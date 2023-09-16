"""
Configuration in this file ared used in preprocessing, training and postprocessing stages
"""

import platform
import os.path

isPreprocessing = False          # False: for training only, skip preprocessing on Windows
testing = False   #  True only for data preprocessing debugging purpose, test data not uploaded to repo

dataset_name = "FreeCAD_lib"    # Mechanical CAD part library
#dataset_name = "KiCAD_lib"      # ECAD KiCAD packaging library
#dataset_name = "Thingi10K"      # all data in one folder, do not use, categorization not ideal
#dataset_name =  "ModelNet10"    #  has two variants, ModelNet10 and ModelNet40
#dataset_name =  "ModelNet40"
#dataset_name =  "ShapeNetCore"  # this work is done in workplace PC, source code not available

########################################################################
# try find preprocessed image and metadata in this repo, without preprocessing
INPUT_DATA_DIR = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "data" + os.path.sep
DATA_DIR = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "data" + os.path.sep
# if not found, use machine specific path, for developer/anyone want to reproduce this work
is_developer_mode = False
if is_developer_mode:
    if platform.system() == "Linux":
        INPUT_DATA_DIR = "/media/DataDir/"
        DATA_DIR="/media/DataDir/"  # data folder contains the raw 3D model data like FreeCAD-library git repo
    else:  # windows OS
        INPUT_DATA_DIR = "E:/AICAE_DataDir/"
        DATA_DIR = "E:/AICAE_DataDir/"  # used only in DTV generation stage
# in input_parameters and kicad_parameters.py 

input_root_path = DATA_DIR + dataset_name # input folder for preprocessing
output_root_path = DATA_DIR + dataset_name # result folder for preprocessing image data
if is_developer_mode:
    output_root_path = DATA_DIR + dataset_name + "_output"
dataset_dir_path = output_root_path  # output dataset for training liek model checkpoint h5 file
dataset_metadata_filename = dataset_name + "_dataset.json"

########################################################################

isAlreadySplit = False  # split data by subfolder name 'test', 'train', 'validate', dataset_parameter.py will override this if necessary
isModelNet40 = dataset_name == "ModelNet40"
isModelNet40Aligned = True

minimum_sample_count = 20   # will collect 10 groups in freecad_lib if set as 20,  16 groups for 10

## control view generator
generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file, this should be set as True
usingCubeBoundBox = True   #  if True: length, height, width for the boundbox is same length, i.e. shape is not scaled,  this should be set as True
if dataset_name == "KiCAD_lib":
    usingCubeBoundBox = False
usingOBB = False  # use optimum orientation bound (OBB), False if models have been aligned/oriented

usingKerasTuner = False         # tested, not quite useful, so set as False
usingMixedInputModel = True     # False: if use only image input, CNN, no multiple linear parameter submodel
usingMaxViewPooling = False     # False: use image concat, instead of pooling,

# BOP method is very slow (100 times slower for assembly) but more robust, for geometry input only
# set False to use ray-triangulation-projection algorithm designed for this
usingBOP = False


## control data collector padding
paddingImage = False  # let tensorflow do random padding for data augmentation if needed
# preprocessed images should be 60X60, left 4 pixels for random shifting (data augment)
# training input data could be 64X64Xchannels after padding


## control total view count, baseline 3 views
view_count = 3
usingRotatingZAxisView = False # rotate currently rotate Z axis by 45degree, to generate 6 views
usingTriView = False  # not impl yet, so it must be False
if usingRotatingZAxisView:
    view_count = 6

## model variant test, 
# control channel count, baseline is thickness + depthmap 2 channels
usingOnlyThicknessChannel = False  # if False, use thickness and depth
usingOnlyDepthmapChannel = False  # if False, use thickness and depth
# channel 3 could also be used
channel_count = 2 if generatingThicknessViewImage else 1
channel_count = 1 if usingOnlyThicknessChannel or usingOnlyDepthmapChannel else channel_count
depthmap_channel = 0  # first channel
thickness_channel = 1  # second channel
backdepth_channel = 2