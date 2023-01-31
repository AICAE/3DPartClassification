# 3DTV-CNN: Deep learning on 3D CAD geometry classification

DTV-CNN: deep learning on fusion of depth and thickness views for 3D shape classification for CAD software

Qingfeng Xia
https://www.researchgate.net/profile/Qingfeng_Xia

MIT licensed
---

## Innovation
+ new 3D geometry datasets: from prominent open source CAD projectss FreeCAD and KiCAD
+ new preprocessor: thickness map 2.5D image
+ information entropy: faster for better preprocessor for more info per image
+ new deep learning model for 3D geometry from realistic CAD software:  
+ explore new application of 3D classification to enable digital engineering

## Literature view of existing models

see another md doc

## Prerequisits

This method does not requrie GPU to complete the training, laptop CPU is fine

### Tested platforms
The whole workflow has currently tested on Ubuntu 18.04/20.04 only, while it should work on windows, just taking time to sort out C++ building dependencies. OpenCV FreeCAD, and OpenCASCADE C++ dev env should be installed, which is troublesoome on Windows.  

Windows users can download the preprocessed data in numpy file format. 

`pip install pydot graphviz python3-opencv scikit-learn tensorflow`

Install graphviz executable (make sure dot executable is on path) from official website, and then `pip install pydot graphviz` to plot tensorflow model.



### Other models
AICAE/VoxNet-Tensorflow-V2: migrate VoxNet-Tensorflow to Tensorflow V2 API (github.com)

"3D shape classification and retrieval based on polar view"
"Meshnet: Mesh neural network for 3d shape representation"

https://github.com/chrischoy/3D-R2N2 3D dense voxel , 

(7859, 3, 60, 60, 2) modelnet40

## Data preprocessing and training workflow

The whole workflow has currently tested on Ubuntu only, while it should work on windows.  Windows user can download the numpy.array images + pandas metadata file, without preprocessing raw data (step 1 to 3 below), 

1. Configuration: `global_config.py` select data source and saved dataset file names,  also set `isPreprocessing`
   `input_parameters.py` contains data source specific setup parameters
    
2. Preprocessing: `dataGenerator.py`  
   generate classification data from folder structure into a single json file. 
   This script use scripts below:
   + `partConverter.py`: use FreeCAD python API to convert step into brep, to feed ViewGenerators app `OccQt`: native executables that generate geometry metadata and dump views into images. 
   + `meshPreprocessor.py`:  using FreeCAD python API to convert mesh file format into stl, to feed thickness view generator app `OccProjector`
   + `imagePreprocessor.py`: image crop, channel fusion for thickness and depth view

3. Prepare Dataset (collecting)
   `sudo apt install python3-tqdm python3-pandas python3-opencv`
   + `dataCollector.py`  generate numpy file containing the images with corresponding single json file metadata
   + `dataSummary.py`  stat data and overview classification
   resize and merge images (opencv2) into numpy.array + json meta files into pandas DF

===
   Note: ViewGenerators (OccQt/OccProjector) is based on OpenCACADE C++ API, no doc is provided for build on Windows. If you have FreeCAD installed and with the FreeCAD development env setup, you should be enable to compile it.

   OccQt and OccProjector are subproject, with cmake to assist compiling. 
===   

4. Traing: `partClassify.py`: TensorFlow model mixed data (images, category data)
  `DTVmodel.py`, `stratify.py`

5. Postprocessing: `plotModel.py`



## Dataset

### split for training data and testing data: 
+ for FreeCAD and KiCAD lib dataset, using `stratify.py` to extract 1 sample from 5 samples as test data. 
+ for other dataset, train and test data are in different subfolder
```py
# in my_split()
for i, v in group_data["subcategories"].iteritems():
      if train_folder_name in v:
         train_g.append(i)
      else:
         test_g.append(i)

# in partClassify.py
if dataset_name.find("ModelNet") >= 0 or dataset_name.find("ShapeNet") >= 0:
    train_folder_name="train"
else:
    train_folder_name=None  # by a 80% ans 20% split
```

### Tensorflow has ShapeNet ModelNet40

https://www.tensorflow.org/graphics/api_docs/python/tfg/datasets/shapenet/Shapenet

Aligined ModelNet40 dataset
https://github.com/lmb-freiburg/orion

### ShapeNetCore

### FreeCAD library dataset
filter out category that has too smaller item. Actually, it has been done auto by tensorflow
> WARN: group size is too small to split, skip this group

fc_lib has only 3 groups for mechdata

`tensorboard --logdir logs/scalars`  will plot the epoch_loss 

Fasterners has 2 screws folders have been merged. (pull Feb 05, 2021)


### CAD-CAP: a 25,000-image database serving the development of artificial intelligence for capsule endoscopy

### Drexel CAD: no download link available
model Datasets (http://edge.cs.drexel.edu/repository/)

### Electronics 3D parts from KiCad project

https://github.com/KiCad/kicad-packages3D/tree/master/Crystal.3dshapes
https://kicad.github.io/packages3d/

STEP and WRL
WRL files are an extension of the Virtual Reality Modeling Language (VRML) format .

why? classification and match to setup simulation


## Comparison study

see also <docs/Todo.md>
### compressed image, vs not compressed

bit compression, 

6 views for modelnet10,  maxcube, 
non-watertight may cause some problem
11/11 [==============================] - 31s 3s/step - loss: 0.1789 - accuracy: 0.9412 - val_loss: 0.2128 - val_accuracy: 0.9392


3 views as a group?
per class error matrix plot

converted part: /media/qxia/QingfengXia/AICAE_DataDir/ModelNet40_output_thickness/airplane/test/airplane_0683.stl
face_normals contain NaN, ignoring!
face_normals contain NaN, ignoring!

converted part: /media/qxia/QingfengXia/AICAE_DataDir/ModelNet40_output_thickness/airplane/test/airplane_0687.stl
face_normals contain NaN, ignoring!

volume data is not correct!
cone                         0.005735            0.700597            0.743134          1.010104e-02        1.010107e-02          1.010102e-02

### different image resolution, 

quick enough by threading

image pixel is not fixed, why?

scale to aspect ratio 1:1

### using geometry meta data only or images only


