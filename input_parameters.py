from __future__ import print_function, division
import os
import os.path
import json
import subprocess
import shutil
import glob

testing = False   # for debugging purpose
usingMixedInputModel = True  # False: if use only image input 

generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file
channel_count = 3 if generatingThicknessViewImage else 1
usingOnlyThicknessChannel = True
channel_count = 1 if usingOnlyThicknessChannel else channel_count

generatingMultiViewImage = not generatingThicknessViewImage
usingGrayscaleImage = not generatingThicknessViewImage
# also generate meta data for CAD geometry like step file

#datasetName = "Thingi10K"       # all data in one folder
datasetName =  "ModelNet"       # 
#datasetName = "fclib"    

metadata_suffix = "json"
hasPerfileMetadata  = False  #  detected by replace input file suffix to json and test existence
# but still needs to merge extra

if datasetName == "Thingi10K":

    isMeshFile = True    # choose between  part and mesh input format
    hasPerfileMetadata  = True

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
    # off mesh file is not manifold, cause error in thickness view generation
    hasPerfileMetadata  = False  # where is the metadata like tag and c

    ##############################
    if testing:
        root_path = "./testdata/testModelNet_data"
        output_root_path = "./testdata/testModelNet_output"
        dataset_metadata_filename = "testModelNet_dataset.json"
    else:
        root_path = "/mnt/windata/MyData/ModelNet10"
        output_root_path = "/opt/ModelNet10_output"
        dataset_metadata_filename = "ModelNet10_dataset.json"

elif datasetName == "fclib":
    from fclib_parameters import *
    isMeshFile = False
else:
    print(datasetName, "dataset not supported, check spelling")

if generatingThicknessViewImage:
    output_root_path = output_root_path + "_thickness"

##########################
# preprocessing  meshlib meshed part dataset_metadata_filename
##########################
if isMeshFile:
    supported_input_file_suffices = set(["off", "stl"])  #  freecad/meshio support a few mesh format, off
    input_file_suffix = "stl"
    # stl is the only format needed for view generator

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
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

############## image preprocessing ############
binarizingImage = not generatingThicknessViewImage
compressingImage = False and binarizingImage
concatingImage = True # do not concat, so image can be flipped

## image pixel size has been hardcoded into view image generator apps
if generatingThicknessViewImage:
    normalizingImage = True
    im_width, im_height = 64, 64   # generated
    IM_WIDTH, IM_HEIGHT = 32, 32   # after padding, for input into tensorflow
else:
    if datasetName == "Thingi10K":
        im_width, im_height = 60, 60   # before padding
        #final image size to be learned by AI
        IM_WIDTH, IM_HEIGHT = 64, 64
    else:
        im_width, im_height = 120, 120   # before padding
        # final image size to be learned by AI
        IM_WIDTH, IM_HEIGHT = 128, 128

        ## compression  only for some CAD part
        compressingImage = False #  compressed or not compressed, both working
        block_size = 8  # can be 4,  4X4 binary pixels compressed into uint16

paddingImage = (IM_WIDTH > im_width) or (IM_HEIGHT > im_height)
if compressingImage:
    result_shape = (IM_HEIGHT//block_size, IM_WIDTH//block_size,  view_count)
else:
    result_shape = (IM_HEIGHT, IM_WIDTH, view_count)  # Y axis as the first index in matrix data

## concat
if concatingImage:
    result_shape = [result_shape[0], result_shape[1]*view_count, channel_count]
else:
    result_shape = [view_count, result_shape[0], result_shape[1], channel_count]

# if channel_count > 1:
#     result_shape.append(channel_count)
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

## saved model file to continue model fit
_saved_model_file = "model_saved"
if usingMixedInputModel:
    _saved_model_file = "mixed_input_" +  _saved_model_file

if generatingMultiViewImage:
    _saved_model_file = "multiview_" +  _saved_model_file
else:
    if usingOnlyThicknessChannel:
        _saved_model_file = "thickness_" +  _saved_model_file
    else:
        _saved_model_file = "3ch_" +  _saved_model_file

saved_model_file = output_root_path + os.path.sep + _saved_model_file