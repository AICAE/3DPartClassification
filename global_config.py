INPUT_DATA_DIR = "/media/qxia/QingfengXia/AICAE_DataDir/"
DATA_DIR="/mnt/windata/DataDir/"

testing = False   # for debugging purpose
usingMixedInputModel = True # False: if use only image input
usingMaxViewPooling = False # False: use concat

#dataset_name = "Thingi10K"       # all data in one folder, do not use, categorization not ideal
dataset_name =  "ModelNet"       #  has two variants, modelnet10 and modelnet40
isModelNet40 = False
#dataset_name = "FreeCAD_lib"   # Mechanical CAD part library
#dataset_name = "KiCAD_lib"       # ECAD KiCAD library

usingCubeBoundBox = False
usingOBB = False
usingXYZview = False

usingKerasTuner = False

generatingThicknessViewImage = True # also generate meta data for CAD geometry like step file
usingOnlyThicknessChannel = False  # if False, use thickness and depth
channel_count = 2 if generatingThicknessViewImage else 1
channel_count = 1 if usingOnlyThicknessChannel else channel_count
thickness_channel = 1  # second channel