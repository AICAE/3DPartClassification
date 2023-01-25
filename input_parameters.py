# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import os.path
import json
import subprocess
import shutil
import sys
import glob

from global_config import *

generatingMultiViewImage = not generatingThicknessViewImage
# generatingMultiViewImage to eval/repeat MVCNN is deprecated
usingGrayscaleImage = not generatingThicknessViewImage
# also generate meta data for CAD geometry like step file


metadata_suffix = "json"
hasPerfileMetadata  = False  #  detected by replace input file suffix to json and test existence
# but still needs to merge extra
isValidSubfolder = lambda dir: True   # dump subfolder filter, to be overidden if necessary

if dataset_name == "Thingi10K":

    isMeshFile = True    # choose between  part and mesh input format
    hasPerfileMetadata  = True

    ##############################
    if testing:
        input_root_path = "./testdata/testThingi10K_data"
        output_root_path = input_root_path + "_output"
        dataset_metadata_filename =  "testThingi10K_dataset.json"
    else:
        input_root_path = DATA_DIR + "Thingi10K_dataset"
        output_root_path = DATA_DIR + "Thingi10K_dataset_output"
        dataset_metadata_filename = "Thingi10K_dataset.json"

    def collect_metadata():
        # only works if all input files are in the same folder
        metadata = {}
        if hasPerfileMetadata:
            # no needed for input format conversion
            all_input_files = glob.glob(input_root_path  + os.path.sep  + "*." + metadata_suffix)
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
elif dataset_name.find("ModelNet") >= 0:
    isMeshFile = True    # choose between  part and mesh input format
    # off mesh file is not manifold, cause error in thickness view generation
    hasPerfileMetadata  = False  # where is the metadata like tag and c

    ##############################
    if testing:
        input_root_path = "./testdata/testModelNet_data"
        output_root_path = input_root_path  + "_output"
        dataset_metadata_filename = "testModelNet_dataset.json"
    else:
        if isModelNet40:
            if isModelNet40Aligned:
                dataset_full_name = "ModelNet40Aligned"
            else:
                dataset_full_name = "ModelNet40"
            input_root_path = INPUT_DATA_DIR + dataset_full_name
            output_root_path = input_root_path + "_output"
            dataset_dir_path = DATA_DIR + dataset_full_name + "_output"
            dataset_metadata_filename = dataset_full_name + "_dataset.json"
        else:
            input_root_path = DATA_DIR + "ModelNet10"
            output_root_path = DATA_DIR + "ModelNet10_output"
            dataset_dir_path = output_root_path
            dataset_metadata_filename = "ModelNet10_dataset.json"

elif dataset_name.find("ShapeNetCore") >= 0:
    isMeshFile = True    # choose between  part and mesh input format
    # off mesh file is not manifold, cause error in thickness view generation
    hasPerfileMetadata  = False  # where is the metadata like tag and c

    ##############################
    if testing:
        input_root_path = "./testdata/ShapeNetCore_data"
        output_root_path = input_root_path  + "_output"
        dataset_metadata_filename = "testShapeNetCore_dataset.json"
    else:
        input_root_path = DATA_DIR + "ShapeNetCore"
        output_root_path = DATA_DIR + "ShapeNetCore_output"
        dataset_dir_path = output_root_path
        dataset_metadata_filename = "ShapeNetCore_dataset.json"

elif dataset_name == "KiCAD_lib":
    from kicad_parameters import *

elif dataset_name == "FreeCAD_lib":
    from fclib_parameters import *
    isMeshFile = False
else:
    print(dataset_name, "dataset not supported, check spelling")

if generatingThicknessViewImage:
    output_root_path = output_root_path + "_thickness"
    dataset_dir_path = dataset_dir_path + "_thickness"

##########################
# preprocessing  meshlib meshed part dataset_metadata_filename
##########################
if isMeshFile and isPreprocessing:
    supported_input_file_suffices = set(["off", "stl", "obj"])  #  freecad/meshio support a few mesh format, off
    # obj mesh format should be covnerted into stl by meshio
    input_file_suffix = "stl"
    # stl is the only format needed for view generator

    # freecad output stl can not been read by occt, but we need freecad to calc bbox ,etc
    import fcMeshPreprocessor
    # return a dict of bbox, area, volume,
    def generate_metadata(input, json_file_path):
        info = fcMeshPreprocessor.generateMetadata(input)
        with open(json_file_path, "w") as outfile:
            json.dump(info, outfile)
            print("generated metadata file: ", json_file_path)
        return info

    # meshio, MeshLab  can also convert mesh from .off to .stl
    def convert(input, output):
        cmd = ["meshio-convert", input, output]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, error = p.communicate()
        if p.returncode != 0:
            print("failed to convert ", input, str(out), str(error))
        return p.returncode == 0

#if isPreprocessing:
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

############## image preprocessing ############
binarizingImage = not generatingThicknessViewImage
compressingImage = False and binarizingImage
concatingImage = False # do not concat !!!, so image can be flipped, and have view pooling


## image pixel size has been hardcoded into view image generator apps
if generatingThicknessViewImage:
    normalizingImage = True
    if usingCubeBoundBox:
        im_width, im_height = 60, 60   # generated
    else:
        im_width, im_height = 60, 60   # generated
    if paddingImage:
        model_input_width, model_input_height = 64, 64   # after padding, for input into tensorflow
    else:
        model_input_width, model_input_height = im_width, im_height
else:
    im_width, im_height = 120, 120   # before padding
    # final image size to be learned by AI
    model_input_width, model_input_height = 128, 128

    ## compression  only for some CAD part
    compressingImage = False #  compressed or not compressed, both working
    block_size = 8  # can be 4,  4X4 binary pixels compressed into uint16

if compressingImage:
    result_shape = (view_count, model_input_height//block_size, model_input_width//block_size)
else:
    result_shape = (view_count, model_input_height, model_input_width)  # Y axis as the first index in matrix data

#paddingImage = (model_input_width > im_width  and model_input_width < im_width* 1.5) or (model_input_height > im_height)
## concat
if concatingImage:
    model_input_shape = [model_input_height, model_input_width*view_count, channel_count]
else:
    model_input_shape = [view_count, model_input_height, model_input_width, channel_count]

#########################################################################################
########### dataset save control #########
if testing:
    dataset_dir_path = output_root_path
    if os.path.exists(output_root_path):
        os.system("rm -rf {}".format(output_root_path))  # this is not portable, posix only

if not os.path.exists(dataset_dir_path):
    os.makedirs(dataset_dir_path)
# it is better to output to another folder but keep folder structure, for easy clean up
dataset_metadata_filepath = output_root_path + os.path.sep + dataset_metadata_filename


_processed_imagedata_filename = dataset_name + "_imagedata.npy"

if not concatingImage:
    _processed_imagedata_filename = str(view_count) + "view_" + _processed_imagedata_filename

if usingCubeBoundBox:
    _processed_imagedata_filename = "cubebox_" + _processed_imagedata_filename

if compressingImage:
    _processed_imagedata_filename = "compressed_" + _processed_imagedata_filename
else:
    _processed_imagedata_filename = "processed_" + _processed_imagedata_filename

processed_imagedata_filepath = dataset_dir_path + os.path.sep + _processed_imagedata_filename
_processed_metadata_filename = _processed_imagedata_filename[:-14] + "_metadata.json"
processed_metadata_filepath = dataset_dir_path + os.path.sep + _processed_metadata_filename
######################################################################################
## saved model file to continue model fit
_saved_model_name = dataset_name
if dataset_name.find("ModelNet") >= 0:
    if isModelNet40:
        _saved_model_name = "ModelNet40"
    else:
        _saved_model_name = "ModelNet10"

if usingMixedInputModel:
    _saved_model_name = "mixedinput_" +  _saved_model_name
else:
    _saved_model_name = "imageonly_" +  _saved_model_name
if usingCubeBoundBox:
    _saved_model_name = "cubebox_" +  _saved_model_name
if usingMaxViewPooling:
    _saved_model_name = "maxviewpooling_" +  _saved_model_name

if view_count != 3:
    _saved_model_name = str(view_count) + "View_" +  _saved_model_name

if generatingMultiViewImage:  # MVCNN
    _saved_model_name = "multiview_" +  _saved_model_name
else:
    if usingOnlyThicknessChannel:
        _saved_model_name = "thickness_" +  _saved_model_name
    elif usingOnlyDepthmapChannel:
        _saved_model_name = "depthmap_" +  _saved_model_name
    else:
        _saved_model_name = "DT_" +  _saved_model_name

saved_model_filepath = dataset_dir_path + os.path.sep + _saved_model_name + ".h5"


modelnet40_classes = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']