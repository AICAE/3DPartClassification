import platform
import os.path

# try find preprocessed image and metadata in this repo, without preprocessing
DATA_DIR = os.path.abspath(os.path.dirname(__file__)) +os.path.sep + "DataDir"
# if not found, use machine specific path, for developer
if not os.path.exists(DATA_DIR):
    if platform.system() == "Linux":
        INPUT_DATA_DIR = "/mnt/windata/DataDir/"
        DATA_DIR="/mnt/windata/DataDir/"
    else:
        INPUT_DATA_DIR = "D:/DataDir/"
        DATA_DIR="D:/DataDir/"

isPreprocessing = False          # False: for training only

#dataset_name = "Thingi10K"      # all data in one folder, do not use, categorization not ideal
dataset_name =  "ModelNet"       #  has two variants, modelnet10 and modelnet40
isModelNet40 = False
isModelNet40Aligned = False
#dataset_name = "FreeCAD_lib"    # Mechanical CAD part library
#dataset_name = "KiCAD_lib"      # ECAD KiCAD library
minimum_sample_count = 20

usingKerasTuner = False         # tested, not quite useful
testing = False   #  True only for data preprocessing debugging purpose, using test data
usingMixedInputModel = True     # False: if use only image input, CNN, no parameter
usingMaxViewPooling = False      # False: use concat

# control view generator
generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file
usingCubeBoundBox = True   #  length, height, width for the boundbox is same length, i.e. shape is not scaled
usingOBB = False  # use orientation bound box
usingBOP = True  # very slow but more robust, for geometry input only

# control total view count, baseline 3 views
view_count = 3
usingRotatingZAxisView = True # rotate currently rotate Z axis by 45degree, using 6 views
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