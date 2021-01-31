from __future__ import print_function, division
import os
import os.path

##########################
# FreeCAD standard part library
#########################
# download part data and generate classification by folder structure

testing = False
MechanicalOnly = True  # no electronics parts
splittingFastenerCategory = True
# can be further limited to fasteners of Mechanical parts

# lower(), some fcstd may does not have step corresponding, some may have only step
supported_input_file_suffices = set(["stp", "step"])  # "FCStd"
input_file_suffix = "brep"

from partConverter import convert

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
    output_root_path = "/mnt/windata/MyData/freecad_library_output"
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    if MechanicalOnly:
        input_root_path = "/mnt/windata/MyData/FreeCAD-library/Mechanical Parts"
        dataset_metadata_filename = "mechdata.json"
    else:
        input_root_path = "/mnt/windata/MyData/FreeCAD-library"
        dataset_metadata_filename = "alldata.json"


####################  should be shared ##############################
