from __future__ import print_function, division
import os
import os.path


from global_config import *
##########################
# FreeCAD standard part library
#########################
# download part data and generate classification by folder structure

testing = False
MechanicalOnly = True  # no electronics parts
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

###########################
# generate image by python + command line program written in C++
#if generatingMultiViewImage:
displayingMode = "Shaded"  # "WireFrame"
# input image  size dumped from 3D views
image_width = 480
image_height = 480


################# IO path #################
if testing:
    input_root_path = "./testdata/testFreeCAD_lib"
    output_root_path = input_root_path + "_output"
    dataset_metadata_filename =  output_root_path + "/test_data.json"
else:
    output_root_path = DATA_DIR + "freecad_library_output"
    dataset_dir_path = DATA_DIR + "freecad_library_output"

    if MechanicalOnly:
        input_root_path = DATA_DIR + "FreeCAD-library/Mechanical Parts"
        dataset_metadata_filename = "mechdata.json"
    else:
        input_root_path = DATA_DIR + "/FreeCAD-library"
        dataset_metadata_filename = "alldata.json"


####################  should be shared ##############################
