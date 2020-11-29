from __future__ import print_function, division
import os
import os.path

##########################
# FreeCAD standard part library
#########################
# download part data and generate classification by folder structure

testing = False
MechanicalOnly = True  # no electronics parts
# can be further limited to fasteners of Mechanical parts

# lower(), some fcstd may does not have step corresponding, some may have only step
supported_input_file_suffices = set(["stp", "step"])  # "FCStd"
input_file_suffix = "brep"

from partConverter import convert

###########################
# generate image by python + command line program written in C++
displayingMode = "Shaded"  # "WireFrame"
# input image  size dumped from 3D views
image_width = 480
image_height = 480


################# IO path #################
if testing:
    root_path = "./testdata/testdata_fclib"
    output_root_path = "./testdata/testdata_fclib_output"
    dataset_filename =  "./testdata/testdata_fclib_output/test_data.json"
else:
    output_root_path = "/opt/freecad_library_output"
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    if MechanicalOnly:
        root_path = "/opt/FreeCAD-library/Mechanical Parts"
        dataset_filename = "mechdata.json"
    else:
        root_path = "/opt/FreeCAD-library"
        dataset_filename = "alldata.json"


####################  should be shared ##############################
