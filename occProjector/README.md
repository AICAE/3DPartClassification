
## Thickness view generation algorithm

by Qingfeng Xia


## Installation

OpenCASCADE, OpenCV

using CMake to build  the project, c++17 is needed to compile, argparse.hpp must use C++17. 


## 3D data source

### Specialty of CAD 3D object 
+ CAD only planar surface, surface types
+ no texture, material
+ precise,  volume, key dimension
standard part


### **ModelNet (2015)** 
<http://modelnet.cs.princeton.edu/#>
127915 3D CAD models from 662 categories
ModelNet10: 4899 models from 10 categories  OFF mesh format , not CAD format
ModelNet40: 12311 models from 40 categories, all are uniformly orientated


ModelNet10  OFF data  can not loaded into FreeCAD, then it is fixed. 
https://people.sc.fsu.edu/~jburkardt/examples/meshlab/meshlab.html

[![img](https://camo.githubusercontent.com/81660f0663d79e35760d49c46ca08a761a39bc4b3f209d243c30801207286e72/687474703a2f2f3364766973696f6e2e7072696e6365746f6e2e6564752f70726f6a656374732f323031342f4d6f64656c4e65742f7468756d626e61696c2e6a7067)](https://camo.githubusercontent.com/81660f0663d79e35760d49c46ca08a761a39bc4b3f209d243c30801207286e72/687474703a2f2f3364766973696f6e2e7072696e6365746f6e2e6564752f70726f6a656374732f323031342f4d6f64656c4e65742f7468756d626e61696c2e6a7067)

**Thingi10K: A Dataset of 10,000 3D-Printing Models (2016)** [[Link\]](https://ten-thousand-models.appspot.com/)
10,000 models from featured “things” on thingiverse.com, suitable for testing 3D printing techniques such as structural analysis , shape optimization, or solid geometry operations.

9GB  stl raw meshes

Total number of files: 10,000

STL files: 9956
OBJ files: 42
PLY files: 1
OFF files: 1



[![img](https://camo.githubusercontent.com/8a53af3b713b081bf6731951b556037bbb7f88bd146a5d3383121662638d7f3e/68747470733a2f2f7062732e7477696d672e636f6d2f6d656469612f44526278576e71586b4145454830672e6a70673a6c61726765)](https://camo.githubusercontent.com/8a53af3b713b081bf6731951b556037bbb7f88bd146a5d3383121662638d7f3e/68747470733a2f2f7062732e7477696d672e636f6d2f6d656469612f44526278576e71586b4145454830672e6a70673a6c61726765)



**ABC: A Big CAD Model Dataset For Geometric Deep Learning** [[Link\]](https://cs.nyu.edu/~zhongshi/publication/abc-dataset/)[[Paper\]](https://arxiv.org/abs/1812.06216)
This work introduce a dataset for geometric deep learning consisting of over 1 million individual (and high quality) geometric models, each associated with accurate ground truth information on the decomposition into patches, explicit sharp feature annotations, and analytic differential properties.

[![img](https://camo.githubusercontent.com/74d05e828ae6378e3f84f4ef9b4cf30937c5ffa282032a00b69c84eefdd62498/68747470733a2f2f63732e6e79752e6564752f7e7a686f6e677368692f696d672f6162632d646174617365742e706e67)](https://camo.githubusercontent.com/74d05e828ae6378e3f84f4ef9b4cf30937c5ffa282032a00b69c84eefdd62498/68747470733a2f2f63732e6e79752e6564752f7e7a686f6e677368692f696d672f6162632d646174617365742e706e67)





## 3D to 2D image projection approaches

Sihouette image, 
depth image  (hollow shape)
xray, thickness

### multiple view by occQt, or just FreeCAD

MVCNN use 12 views. 
needs 6 views for color rendering


###  generate multiple views

ShapeNet
ModelNet
https://github.com/Chinmay26/Multi-Viewpoint-Image-generation
java + scala render

### volumetric reader and generation

voxbin

https://www.patrickmin.com/binvox/      output VTK format
https://www.patrickmin.com/viewvox/

surface mesh voxelizer: <https://github.com/karimnaaji/voxelizer>

FreeCAD: https://forum.freecadweb.org/viewtopic.php?t=30172


### x-ray volumetric shader
blender swap
light absorption
occt project

"xray" -> "in front" mode
https://blenderartists.org/t/what-happened-to-x-ray-in-2-8/1152471


## Implementation

### Workflow

edge must be zero, to mark void, keep vertex at corner? 
padding

1. Shape orientation & optimal bound box
2. extract geometry proeprty by FreeCAD, OBB, volume, area  
3. projection view into image files
4. collecting images into numpy file for better IO performance



### Orientation algorithm

```
build/occProjector   data/part.stl  --bbox 0 0 0 16 26 35
build/occProjector   data/part.brep --bop 
```

### pseudo code for thickness view geneation
stl chair stl got error
got 7 numbers for some pixel of chair.stl,   one point got really big. 
interpolation to fix? 

liner scale of the image seems not good, log scale?

only ascii stl are supported  by occt stl reader?
