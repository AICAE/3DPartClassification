##########################
# KiCAD standard package library
#########################
# download part data and generate classification by folder structure

from __future__ import print_function, division
import os
import os.path


from global_config import *

if testing:
    minimum_sample_count = 1  # set in global_config


# lower(), some fcstd may does not have step corresponding, some may have only step
supported_input_file_suffices = set(["stp", "step"])  # "FCStd"
input_file_suffix = "brep"

try:
    from partConverter import convert
except Exception as e:
    print("Warning: failed to import partConverter", e)

merging_SMD_THT = True  # merge sourface mount and through-hole into one category

KiCAD_Categories = ["Connector_Molex", "Connector_Phoenix_MC", 'Connector_Dsub'
'Connector_PinHeader_1', 'Connector_PinSocket_1', 
'Button_Switch_SMD',  'Button_Switch_THT', 'Capacitor_SMD', 'Capacitor_THT', 'Crystal', 
'Inductor_SMD', 'Inductor_THT', 'Diode_THT',  'LED_THT', "Transformer_THT", 
'Package_BGA', 'Package_SO', 'Package_DIP',
'Relay_THT', 'Resistor_THT', 'TerminalBlock_Phoenix', 'Varistor'
]

# 'Varistor'  looks different from resistor THT, 
# Resistor_SMD and capacitor SMD may look similar
# Inductor_SMD and Capacitor_SMD can not been distuished by shape

################# IO path #################
isMeshFile = False
hasPerfileMetadata  = False
supported_input_file_suffices = set(["stp", "step"])
input_file_suffix = "step"
if testing and is_developer_mode:
    input_root_path = "./testdata/testKiCAD_data"
    output_root_path = input_root_path + "_output"
    dataset_dir_path = input_root_path
    dataset_metadata_filename = "testKiCAD_dataset.json"

if is_developer_mode:
    input_root_path = INPUT_DATA_DIR + "kicad-packages3D"
    output_root_path = DATA_DIR + "kicad-packages3D_output"
    dataset_dir_path = INPUT_DATA_DIR + "kicad-packages3D"
    dataset_metadata_filename = "kicad-packages3D_dataset.json"
# else, if the global config for path

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
