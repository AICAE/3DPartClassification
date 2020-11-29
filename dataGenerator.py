import os.path, shutil, os, sys
import glob, stat
#import zmq
import json
import multiprocessing
import subprocess
from collections import OrderedDict


using_threading = False  # false, can help debugging
resumable= False  # carry on work left, Registry can not resume for the time being

from input_parameters import *


if using_threading:
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(os.cpu_count() -1)
    all_futures={}


# json type from dict,  threading.Queue  is an synchronzied data structure
PART_REGISTRY = OrderedDict()
ERROR_REGISTRY = OrderedDict()
nb_processed =0

def get_filename_stem(input_filename):
    return input_filename[:input_filename.rfind(".")] 

def generate_view_images(input_filename, working_dir=None, info=None):
    # input_filename should be an absolute path in the output folder
    # for mesh type input file, need extra info
    input_filename = os.path.abspath(input_filename)
    input_file_stem = get_filename_stem(input_filename).split(os.path.sep)[-1]
    output_filepath_stem = working_dir + os.path.sep + input_file_stem

    if not working_dir:
        working_dir = os.path.dirname(input_filename)
    assert(os.path.exists(working_dir))
    #print(input_filename, working_dir)
    if isMeshFile:
        assert info
        args = ["--bbox"] + [ str(v) for v in info["bbox"]]
        cmd = [ThicknessViewApp, input_filename] + args
        # , get_filename_stem(input_filename)
        print(cmd)
    else:
        cmd = [ThicknessViewApp, input_filename, get_filename_stem(input_filename)]

    p = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    print(output, error)
    # return code check or result file check
    ret = glob.glob(os.path.dirname(get_filename_stem(input_filename) + "*.png"))
    assert(len(ret))
    #sys.exit()  # debugging stop after processing the first file

def generate_output_file_path(file_path):
    # get relative path, then append with 
    rel = os.path.relpath(os.path.abspath(file_path), root_path)
    out = output_root_path + os.path.sep+ rel
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    return out

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_category_names(category_relative_path):
    # generate category meta data from file path, if no tag metadata available
    cl = splitall(category_relative_path)
    return cl

##########################################

def generate_file_registry(file_path):
    #
    head, tail = os.path.split(file_path)
    category_relative_path = os.path.relpath(head, root_path)
    categories = get_category_names(category_relative_path)
    info={"filename":  tail,   # todo, filename without path may be not unique
              "category": categories[0], 
              "path": category_relative_path}
    if len(categories) > 1:
        info["subcategories"] = categories[1:]
    return info


######################################################
def input_file_exists(file_path):
    suffix = file_path.split('.')[-1]
    stem = file_path[:file_path.rfind(".")]
    for s in supported_input_file_suffices:
        if os.path.exists(stem + s):
            return True
    return False

def check_error(fname):
    statinfo = os.stat(fname)
    if statinfo.st_size < 1:
        return "file byte size {}, is too small".format(statinfo.st_size)

def process_error(filename):
    # can only be run in serial mode
    ERROR_REGISTRY[filename] = check_error(filename)
    print("Delete file `{}` from registry for error".format(filename))
    if not hasPerfileMetadata:
        del PART_REGISTRY[filename]


def _process_input_file(input_file_path, output_file_path):
    json_file_path = output_file_path[:output_file_path.rfind(".")]  + "_metadata.json"
    info = generate_metadata(input_file_path, json_file_path)
    input_metadata_file_path = input_file_path.replace(input_file_suffix, metadata_suffix)
    #hasPerfileMetadata = os.path.exists(input_metadata_file_path)

    convertingInput = not input_file_path.endswith(input_file_suffix) 
    if convertingInput:
        convert(input_file_path, output_file_path)  # if conversion failed, do not register file
        print("converted part:", output_file_path)
        input_file = output_file_path
    else:
        input_file = input_file_path

    if generatingMultiViewImage:
        pass
    if generatingThicknessViewImage:
        generate_view_images(input_file, os.path.dirname(output_file_path), info)
    return True

def process_input_file(input_file_path, output_file_path=None):
    # this function will be run by thread pool executor
    if not output_file_path:
        output_file_path = generate_output_file_path(input_file_path)
    if using_threading:
        try:
            _process_input_file(input_file_path, output_file_path)
        except:
            print("Error in converting or viewing part:", output_file_path)
            return False
    else:
        _process_input_file(input_file_path, output_file_path)

def process_folder(input_folder, output_folder, level=0):
    #
    global nb_processed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    #if level>=5: return None
    subFolders = [o for o in os.listdir(input_folder) if os.path.isdir(input_folder + os.path.sep + o)]
    for d in sorted(subFolders):  #
        input_dir = input_folder + os.path.sep + d
        output_dir = output_folder + os.path.sep + d
        process_folder(input_dir, output_dir, level+1)

    files = [o for o in os.listdir(input_folder) if os.path.isfile(input_folder + os.path.sep + o)]
    for f in files:
        input_file= input_folder + os.path.sep + f
        suffix = (f.split('.')[-1]).lower()
        processed_file_path = output_folder + os.path.sep + f[:f.rfind(".")] + "." + input_file_suffix
        if (suffix in supported_input_file_suffices):
            if not os.path.exists(processed_file_path):
                nb_processed +=1
                if using_threading:
                    future = executor.submit(process_input_file, input_file, processed_file_path)
                    all_futures[input_file] = future
                    # todo: error check
                else:
                    ret = process_input_file(input_file, processed_file_path)
                    if not ret:
                        process_error(input_file)
            if not hasPerfileMetadata:
                # data racing, this must be run in the main thread
                PART_REGISTRY[input_file] = generate_file_registry(input_file)

        elif suffix == "fcstd" and not input_file_exists(input_file):
            #register_file(input_file) # data racing, must be run in main thread
            #process_fcstd_file(fullname, output_file_path)  
            # FCStd is too complicated to  select and convert
            pass
        elif suffix== "brep":
            print("there should be no brep file in library")
        else:
            pass

"""
def process_folder_serial(folder_path):
    # what is the difference?     threading is not used in this function
    for rootfolder, folders, files in os.walk(folder_path):
        for file in files:
            fullname = rootfolder + os.path.sep + file
            suffix = (fullname.split('.')[-1]).lower()
            if (suffix in supported_input_file_suffices):
                register_file(fullname)
                if convertingInput:
                    process_input_file(fullname)
            elif suffix == "fcstd" and not input_file_exists(fullname):
                #process_fcstd_file(fullname)  FCStd is too complicated to  select and convert
                pass
            elif suffix== "brep":
                print("there should be no brep file in library")
            else:
                pass
        for folder in folders:
            print("processing folder: ", folder)
            fullname = rootfolder + os.path.sep + folder
            process_folder_serial(fullname)
"""

def write_dataset_metadata(dataset_filename):
    if hasPerfileMetadata:
        dataset_metadata = collect_metadata()
    else:
        dataset_metadata = list(PART_REGISTRY.values())  # list is not as good as dict
    with open(dataset_filename, 'w') as f:
        json.dump(dataset_metadata, f, indent=4)
        print("data has been write into file: ", dataset_filename)


##################################
if __name__ == "__main__":
    process_folder(root_path, output_root_path)
    #process_folder1(root_path)
    if using_threading:
        #wait all, another way is executor.shutdown()
        for file, fu in all_futures.items():
            if not fu.result():
                process_error(file)
    write_dataset_metadata(dataset_filename)
    print("total registered files = ", len(PART_REGISTRY))
    print("total processed files = ", nb_processed)