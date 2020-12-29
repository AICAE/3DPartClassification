from __future__ import print_function, division
import os
import os.path
import json
import subprocess
import shutil
import glob

generatingThicknessViewImage=True # also generate meta data for CAD geometry like step file
generatingMultiViewImage=False # also generate meta data for CAD geometry like step file
datasetName = "Thingi10K"       # all data in one folder
#datasetName =  "ModelNet" not usable dataset !
#datasetName = "fclib"    

metadata_suffix = "json"
hasPerfileMetadata  = False  #  detected by replace input file suffix to json and test existence
# but still needs to merge extra

if datasetName == "Thingi10K":

    isMeshFile = True    # choose between  part and mesh input format
    hasPerfileMetadata  = True
    supported_input_file_suffices = set(["off", "stl"])  #  freecad/meshio support a few mesh format, off
    input_file_suffix = "stl"
    # stl is the format needed for view generator, no need for conversion

    testing = False   # for debugging purpose
    ##############################
    if testing:
        root_path = "./testdata/testThingi10K_data"
        output_root_path = "./testdata/testThingi10K_output"
        dataset_metadata_filename =  "testThingi10K_dataset.json"
    else:
        root_path = "/mnt/windata/MyData/Thingi10K_dataset"
        output_root_path = "/mnt/windata/MyData/Thingi10K_dataset_output"
        dataset_metadata_filename = "Thingi10K_dataset.json"

    def collect_metadata():
        # only works if all input files are in the same folder
        metadata = {}
        if hasPerfileMetadata:
            # no needed for input format conversion
            all_input_files = glob.glob(root_path  + os.path.sep  + "*." + metadata_suffix)
            #print(all_input_files)
        for input_file in all_input_files:
            #print("process metadata for input file", input_file)
            f = input_file.split(os.path.sep)[-1]
            fileid = f[:f.rfind(".")]
            with open(input_file, "r") as inf:
                m = json.load(inf)
            json_file_path = output_root_path + os.path.sep + str(fileid) + "_metadata.json"
            if(os.path.exists(json_file_path)):
                pm = json.load(open(json_file_path, "r"))
                for k, v in pm.items():
                    m[k] = v
                metadata[fileid] = m
            else:
                print("Warning: there is no generated metdata file", json_file_path)
        return metadata

        # collect from  output folder

    ###########################
elif datasetName == "ModelNet":
    isMeshFile = True    # choose between  part and mesh input format
    # off file is not manifold, cause error in thickness view generation
    hasPerfileMetadata  = False  # where is the metadata like tag and c

    testing = True   # for debugging purpose
    ##############################
    if testing:
        root_path = "./testdata/testmesh_data"
        output_root_path = "./testdata/testmesh_output"
        dataset_metadata_filename =  output_root_path + os.path.sep + "testmesh_data.json"
    else:
        output_root_path = "/opt/ModelNet10_output"
        root_path = "/mnt/windata/MyData/ModelNet10"
        dataset_metadata_filename = "ModelNet10_data.json"

else:
    from fclib_parameters import *

##########################
# preprocessing  meshlib meshed part dataset_metadata_filename
##########################
if isMeshFile:
    # freecad output stl can not been read by occt, but we need freecad to calc bbox ,etc 
    import fcMeshPreprocessor  
    # return a dict of bbox, area, volume,
    def generate_metadata(input, json_file_path):
        info = fcMeshPreprocessor.generateMetadata(input)
        with open(json_file_path, "w") as outfile: 
            json.dump(info, outfile)
            print(json_file_path)
        return info

    # meshio, MeshLab  can also convert mesh from .off to .stl
    def convert(input, output):
        cmd = ["meshio-convert", input, output]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                                    stderr=subprocess.PIPE)
        out, error = p.communicate()
        print(out, error)


###############view app output control ############
# generate image by python + command line program written in C++
ThicknessViewApp = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "occProjector/build/occProjector"
# usage   --bbox
if not os.path.exists(ThicknessViewApp):
    print(ThicknessViewApp, " not found, check your path in your parameter file")
    sys.exit(1)

# can also be a python script
MultiViewApp=os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "occQt/occQt"
# Usage: occQt input_file.brep image_file_stem
# will generate 0.png 1.png 2.png files, not sufficient, in each folder, there may be more than one brep files

##############################
image_suffix = ".png"
view_count = 3
# before random padding the image position

############## image preprocessing ############
if datasetName == "Thingi10K":

    binarizingImage = False
    paddingImage = True
    im_width, im_height = 60, 60   # before padding
    #final image size to be learned by AI
    IM_WIDTH, IM_HEIGHT = 64, 64

    compressingImage = False #  
else:
    # image preprocessing
    binarizingImage = True
    paddingImage = True
    im_width, im_height = 120, 120   # before padding
    #final image size to be learned by AI
    IM_WIDTH, IM_HEIGHT = 128, 128

    ## compression  only for some CAD part
    compressingImage = True  #  compressed or not compressed, both working
    block_size = 8  # can be 4,  4X4 binary pixels compressed into uint16


if compressingImage:
    result_shape = (IM_WIDTH//block_size, IM_HEIGHT//block_size, view_count)
else:
    result_shape = (IM_WIDTH, IM_HEIGHT, view_count)

## concat
concatingImage=True
if concatingImage:
    result_shape = result_shape[0], result_shape[1]*view_count, 1
else:
    result_shape = result_shape[0], result_shape[1], view_count

########### output control #########
if testing:
    if os.path.exists(output_root_path):
        os.system("rm -rf {}".format(output_root_path))  # this is not portable, posix only
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)
# it is better to output to another folder but keep folder structure, for easy clean up
dataset_metadata_filepath = output_root_path + os.path.sep + dataset_metadata_filename

processed_metadata_filepath = output_root_path + os.path.sep + "processed_" + dataset_metadata_filename
if compressingImage:
    processed_imagedata_filename = output_root_path + os.path.sep + "compressed_imagedata.npy"
else:
    processed_imagedata_filename = output_root_path + os.path.sep + "processed_imagedata.npy"