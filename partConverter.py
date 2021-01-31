############################
# it is more efficient to import this module that use it an executable
# FCStd format bearing is too complicated to select (sometime only final visiable is needed) and export

import sys
import os.path

from detect_freecad import append_freecad_mod_path

append_freecad_mod_path()
try:
    import FreeCAD
except ImportError:
    print("freecad is not installed or detectable, exit from this script")
    sys.exit(0)

# version check
import FreeCAD as App
import Part

def convert(input, output):
    """ from step and iges to brep_output
    FCStd objects may be too complicated to convert
    """
    if input.lower().find("fcstd")>0:
        App.openDocument(input)
        default_document_name = App.ActiveDocument.Name
    else:
        default_document_name = "untitled"
        App.newDocument(default_document_name)
        Part.insert(input, default_document_name)
    #obj = doc.ActiveObject   # # single object like brep import
    # step imported assembly could be several docObjects depends on FreeCAD preference
    objs = []
    # todo: recursive inclusion part is not yet supported
    for obj in App.ActiveDocument.RootObjects:
        if obj.isDerivedFrom("Part::Feature"):  # and obj.Visibility
            objs.append(obj)
        if obj.isDerivedFrom("App::Part"):  # for step imported assembly
            objs.append(obj)
        if obj.isDerivedFrom("Part::FeaturePython"): # and obj.Visibility
            objs.append(obj)
        if obj.isDerivedFrom("PartDesign::Part"): # and obj.Visibility
            objs.append(obj)
    Part.export(objs, output)
    del objs
    App.closeDocument(default_document_name)

def test_convert():
    #input = "./testdata/test_screw.fcstd"
    #output = "./testdata/test_screw.step"
    #convert(input, output)
    #assert os.path.exists(output)
    step_input = "./testdata/testFreeCAD_lib_data/test_rod.step"
    brep_output = step_input.replace(".step", ".brep")
    convert(step_input, brep_output)
    assert os.path.exists(brep_output)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: this_script.py input_filename output.filename")
        print("without any parameter files, just run a test")
        test_convert()
    elif len(sys.argv) == 2:
        input = sys.argv[1]
        output = input.replace(".step", ".brep")
        convert(input, output)
    else:
        input = sys.argv[1]
        output = sys.argv[2]
        convert(input, output)