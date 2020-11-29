"""
need FreeCAD daily after Nov 2020
"""
import sys
import os

sys.path.append("/usr/lib/freecad-daily-python3/lib")
# /usr/lib/freecad-daily/Mod
import FreeCAD as App
import Mesh

def convertMesh(input, output):
    Mesh.open(input)

    App.setActiveDocument("Unnamed")
    __objs__ = App.ActiveDocument.Objects

    bbox = __objs__[0].Mesh.BoundBox
    area = __objs__[0].Mesh.Area
    volume = __objs__[0].Mesh.Volume
    #BoundBox (0, 0, 0, 16, 26, 35)   bbox.Xmin, ...

    Mesh.export(__objs__, output)
    del __objs__

    App.closeDocument("Unnamed")
    # from OCCT part,  obb can be calculated 
    # center
    info = {"volume": volume, "area": area, 
        "bbox":  [bbox.XMin, bbox.YMin, bbox.ZMin, bbox.XMax, bbox.YMax, bbox.ZMax]}
    return info

def generateMetadata(input):
    Mesh.open(input)

    App.setActiveDocument("Unnamed")
    __objs__ = App.ActiveDocument.Objects

    bbox = __objs__[0].Mesh.BoundBox
    area = __objs__[0].Mesh.Area
    volume = __objs__[0].Mesh.Volume
    #BoundBox (0, 0, 0, 16, 26, 35)   bbox.Xmin, ...

    #Mesh.export(__objs__, output)
    del __objs__

    App.closeDocument("Unnamed")
    # from OCCT part,  obb can be calculated  center
    info = {"volume": volume, "area": area, 
        "bbox":  [bbox.XMin, bbox.YMin, bbox.ZMin, bbox.XMax, bbox.YMax, bbox.ZMax]}
    return info

#project 
if __name__ == "__main__":
    root_path = "/mnt/windata/MyData/OneDrive/gitrepo/PartRecognition/"
    inputfile =  root_path + u"occProjector/data/mushroom.off"
    output = root_path + u"/occProjector/data/mushroom.stl"
    info = convertMesh(inputfile, output)
    bbox = info["bbox"]
    bbox_arg = " --bbox {} {} {} {} {} {}".format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
    cmd = root_path + u'/occProjector/build/occ_projector  "{}" '.format(inputfile)  + bbox_arg
    print(cmd)
    os.system(cmd)
    print(bbox)

#from plot_views import plot_projection_views
#plot_projection_views(output)