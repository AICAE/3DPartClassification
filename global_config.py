import platform

if platform.system() == "Linux":
    INPUT_DATA_DIR = "/media/qxia/QingfengXia/AICAE_DataDir/"
    DATA_DIR="/mnt/windata/DataDir/"
else:
    INPUT_DATA_DIR = "D:/DataDir/"
    DATA_DIR="D:/DataDir/"

#dataset_name = "Thingi10K"      # all data in one folder, do not use, categorization not ideal
dataset_name =  "ModelNet"       #  has two variants, modelnet10 and modelnet40
isModelNet40 = False
isModelNet40Aligned = False
#dataset_name = "FreeCAD_lib"   # Mechanical CAD part library
#dataset_name = "KiCAD_lib"       # ECAD KiCAD library
minimum_sample_count = 20

testing = False   # for data preprocessing debugging purpose, using test data
usingMixedInputModel = True # False: if use only image input
usingMaxViewPooling = False # False: use concat

# control view generator
usingCubeBoundBox = True   #  length, height, width for the boundbox is same length, i.e. shape is not scaled
usingOBB = False  # use orientation bound box
usingBOP = True  # very slow but more robust, for geometry input only

# 0 or 1 can be enabled
usingRotatingZAxisView = False # rotate currently rotate Z axis by 45degree
usingTriView = False  # not impl yet

usingKerasTuner = False

generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file
usingOnlyThicknessChannel = True  # if False, use thickness and depth
usingOnlyDepthmapChannel = False  # if False, use thickness and depth
channel_count = 2 if generatingThicknessViewImage else 1
channel_count = 1 if usingOnlyThicknessChannel or usingOnlyDepthmapChannel else channel_count
depthmap_channel = 0  # first channel
thickness_channel = 1  # second channel