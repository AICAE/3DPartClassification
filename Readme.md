# 3DTV-CNN: Deep learning on 3D CAD geometry classification

DTV-CNN: deep learning on fusion of depth and thickness views for 3D shape classification for CAD software

Qingfeng Xia
https://www.researchgate.net/profile/Qingfeng_Xia

MIT licensed  2019-2023
---

## Innovation
+ new 3D geometry datasets: from prominent open source CAD projectss FreeCAD and KiCAD
+ new preprocessor: thickness map 2.5D image
+ information entropy: faster for better preprocessor for more info per image
+ new deep learning model for 3D geometry from realistic CAD software:  
+ explore new application of 3D classification to enable digital engineering

## Literature view of existing models

see another md doc, or just see the paper, link to be added later.

## Hardware requirement: CPU is fine

This method does not requrie GPU to complete the training, laptop CPU is sufficient to run the test.

### Tested OS platforms
The whole workflow has currently tested on Ubuntu 18.04/20.04 only, while it should work on windows, just taking time to sort out C++ building dependencies. OpenCV FreeCAD, and OpenCASCADE C++ dev env should be installed, which is troublesoome on Windows.  

Windows users can download the preprocessed data in numpy file format. 

`pip install pydot graphviz python3-opencv scikit-learn tensorflow`

Install graphviz executable (make sure dot executable is on path) from official website, and then `pip install pydot graphviz` to plot tensorflow model.

### Other models compared
AICAE/VoxNet-Tensorflow-V2: migrate VoxNet-Tensorflow to Tensorflow V2 API (AICAE/VoxNet-Tensorflow-V2)
MVCNN:  migrated to Tensorflow V2 API by other developer 

"3D shape classification and retrieval based on polar view"
"Meshnet: Mesh neural network for 3d shape representation"
https://github.com/chrischoy/3D-R2N2 3D dense voxel ,  

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

   `OccQt` and `OccProjector` are subprojects. 
===   

4. Traing: `partClassify.py`: TensorFlow model mixed data (images, category data)
  `DTVmodel.py`, `stratify.py`

5. Postprocessing: `plotModel.py`

### OccProjector : depth-thickness view image generator

It is written in C++ using OpenCASCADE, see the Readme.md in this subproject.

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


## Datasets

This paper has listed all the 3D CAD datasets found, here the preprocessing method for FreeCAD and KiCAD library is described.

### Preproessed dataset are in `data` subfolder

unzip the zip to release numpy file (containing image collection as multi-dimension array), and move npy file wit json file to somewhere


```
processed_3view_KiCAD_lib_imagedata.zip             
processed_3view_KiCAD_lib_metadata.json    
# six views 
processed_nview_FreeCAD_lib_imagedata.zip
processed_nview_FreeCAD_lib_metadata.json

# tensorflow model checkout data
6views_DT_cubebox_mixedinput_ModelNet10_feb12.h5
6views_DT_cubebox_mixedinput_ModelNet10_feb12.h5.json 

processed_cubebox_3view_ModelNet40_metadata.json
processed_cubebox_3view_ModelNet40_imagedata.zip

processed_cubebox_3view_ModelNet10_imagedata.zip
processed_cubebox_3view_ModelNet10_metadata.json 
```

and then edit the INPUT_DATA_DIR and DATA_DIR in  `global_config.py`
```py
if not os.path.exists(DATA_DIR):
    if platform.system() == "Linux":
        INPUT_DATA_DIR = "/media/DataDir/"
        DATA_DIR ="/media/DataDir/"  #output dir
    else:  # windows OS
        INPUT_DATA_DIR = "E:/AICAE_DataDir/"
        DATA_DIR = "E:/AICAE_DataDir/"
```
finally, run the training `partClassify.py`
`kicad_dataset` subfolder, also contains the traing history

### ShapeNet ModelNet40 dataset

https://www.tensorflow.org/graphics/api_docs/python/tfg/datasets/shapenet/Shapenet

Aligined ModelNet40 dataset  https://github.com/lmb-freiburg/orion


### FreeCAD library dataset

hosted on github: https://github.com/FreeCAD/FreeCAD-library
git clone it and edit parameters in `fc_parameter.py` and `dataGenerator.py`

filter out category that has too smaller item. Actually, it has been done auto
> WARN: group size is too small to split, skip this group


### CAD-CAP: a 25,000-image database serving the development of artificial intelligence for capsule endoscopy

### Drexel CAD: no download link available
model Datasets (http://edge.cs.drexel.edu/repository/) is not downloadable

### Electronics 3D parts from KiCad project

github repo: 
https://github.com/KiCad/kicad-packages3D/tree/master/Crystal.3dshapes
https://kicad.github.io/packages3d/

data STEP and WRL
WRL files are an extension of the Virtual Reality Modeling Language (VRML) format .

### ShapeNetCore v2

Code related ShapeNetCore v2 is not published yet, I have modified DTVCNN archicture a bit to improve accuracy. Training was done using computation resource in work place, can not get the code and preprocessed data out. 

### Traceparts and gradcad website

I have tried to download CAD models from <tracparts.com>, but it is not time-assuming even download with webscrape script. 
The half-baken scripts are listed in <tracparts_download>



