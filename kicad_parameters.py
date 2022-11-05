from __future__ import print_function, division
import os
import os.path


from global_config import *
##########################
# FreeCAD standard part library
#########################
# download part data and generate classification by folder structure

testing = False
if testing:
    minimum_sample_count = 1  # set in global_config

splittingFastenerCategory = True
splittingProfileCategory = True
# profile should also be split, but manually
# can be further limited to fasteners of Mechanical parts

# lower(), some fcstd may does not have step corresponding, some may have only step
supported_input_file_suffices = set(["stp", "step"])  # "FCStd"
input_file_suffix = "brep"

try:
    from partConverter import convert
except Exception as e:
    print("Warning: failed to import partConverter", e)

KiCAD_Categories = ["Connector_Molex", "Connector_Phoenix_MC", 'Connector_Dsub'
'Button_Switch_SMD', 'Capacitor_SMD', 'Crystal', 
'Inductor_SMD', 'Diode_THT',  'LED_THT', "Transformer_THT", 
'Package_BGA', 'Package_SO', 'Package_DIP',
'Relay_THT', 'Resistor_THT', 'TerminalBlock_Phoenix'
]

###########################
# generate image by python + command line program written in C++
#if generatingMultiViewImage:
displayingMode = "Shaded"  # "WireFrame"
# input image  size dumped from 3D views
image_width = 480
image_height = 480


################# IO path #################
isMeshFile = False
hasPerfileMetadata  = False
supported_input_file_suffices = set(["stp", "step"])
input_file_suffix = "step"
if testing:
    input_root_path = "./testdata/testKiCAD_data"
    output_root_path = input_root_path + "_output"
    dataset_dir_path = input_root_path
    dataset_metadata_filename = "testKiCAD_dataset.json"
else:
    input_root_path = INPUT_DATA_DIR + "kicad-packages3D"
    output_root_path = INPUT_DATA_DIR + "kicad-packages3D_output"
    dataset_dir_path = DATA_DIR + "kicad-packages3D"
    dataset_metadata_filename = "kicad-packages3D_dataset.json"

def isValidSubfolder(dir):
    if not dir.endswith(".3dshapes"):
        return False
    cat = os.path.basename(dir).split('.')[0]
    if cat not in KiCAD_Categories:
        return False
    if not testing and len(os.listdir(dir)) < minimum_sample_count:
        return False
    print('valid data folder with enough samples ', dir)
    return True

####################  should be shared ##############################
