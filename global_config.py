"""
Configuration in this file ared used in preprocessing, training and postprocessing stages
"""

import platform
import os.path

# try find preprocessed image and metadata in this repo, without preprocessing
DATA_DIR = os.path.abspath(os.path.dirname(__file__)) +os.path.sep + "DataDir"
# if not found, use machine specific path, for developer/anyone want to reproduce this work
if not os.path.exists(DATA_DIR):
    if platform.system() == "Linux":
        INPUT_DATA_DIR = "/media/DataDir/"
        DATA_DIR="/media/DataDir/"  # output dir for checkpoint and history
    else:  # windows OS
        INPUT_DATA_DIR = "E:/AICAE_DataDir/"
        DATA_DIR = "E:/AICAE_DataDir/"  # output dir

# preprocessed images should be 60X60, left 4 pixels for random shifting (data augment)
# training input data could be 64X64Xchannels after padding
isPreprocessing = False          # False: for training only, skip preprocessing on Windows
testing = False   #  True only for data preprocessing debugging purpose, using test data

dataset_name = "FreeCAD_lib"    # Mechanical CAD part library
#dataset_name = "KiCAD_lib"      # ECAD KiCAD library
#dataset_name = "Thingi10K"      # all data in one folder, do not use, categorization not ideal
#dataset_name =  "ModelNet10"    #  has two variants, ModelNet10 and ModelNet40
#dataset_name =  "ModelNet40"
#dataset_name =  "ShapeNetCore"  # this work is done in workplace PC, source code not available

isAlreadySplit = False  # split data by folder name 'test', 'train', 'validate'
isModelNet40 = dataset_name == "ModelNet40"
isModelNet40Aligned = True

minimum_sample_count = 20   # will collect 10 groups in freecad_lib if set as 20,  16 groups for 10

usingKerasTuner = False         # tested, not quite useful
usingMixedInputModel = True     # False: if use only image input, CNN, no multiple linear parameter submodel
usingMaxViewPooling = False     # False: use image concat

# control view generator
generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file
usingCubeBoundBox = False   #  length, height, width for the boundbox is same length, i.e. shape is not scaled
usingOBB = False  # use optimum orientation bound , False if models have been aligned/oriented

# very slow (100 times slower for assembly) but more robust, for geometry input only
# set False to use ray-triangulation-projection algorithm designed for this
usingBOP = False
# control data collector
paddingImage = False  # let tensorflow do random padding for data augmentation if needed

# control total view count, baseline 3 views
view_count = 3
usingRotatingZAxisView = False # rotate currently rotate Z axis by 45degree, to generate 6 views
usingTriView = False  # not impl yet
if usingRotatingZAxisView:
    view_count = 6

# control channel count, baseline is thickness + depthmap 2 channels
usingOnlyThicknessChannel = False  # if False, use thickness and depth
usingOnlyDepthmapChannel = False  # if False, use thickness and depth
channel_count = 2 if generatingThicknessViewImage else 1
channel_count = 1 if usingOnlyThicknessChannel or usingOnlyDepthmapChannel else channel_count
depthmap_channel = 0  # first channel
thickness_channel = 1  # second channel