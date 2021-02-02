


---

## Innovation
+ new 3D dataset
+ new preprocessor: thickness map
+ new deep learning network:  XrayNet
+ faster for better preprocessor , more info per image
+ new application of 3D classification

## Literature view of existing models


see another md doc

## Dataset

### Tensorflow has ShapeNet ModelNet40

https://www.tensorflow.org/graphics/api_docs/python/tfg/datasets/shapenet/Shapenet

Aligined ModelNet40 dataset
https://github.com/lmb-freiburg/orion

#### freecad library dataset
filter out category that has too smaller item. Actually, it has been done auto by tensorflow
> WARN: group size is too small to split, skip this group

fc_lib has only 3 groups for mechdata

`tensorboard --logdir logs/scalars`  will plot the epoch_loss 

### Electronics 3D parts fork KiCad

https://github.com/KiCad/kicad-packages3D/tree/master/Crystal.3dshapes
https://kicad.github.io/packages3d/

STEP and WRL
WRL files are an extension of the Virtual Reality Modeling Language (VRML) format .

why? classification and match to setup simulation



## Workflow

1. `dataSummary.py`  stat data and overview classification
    data downloader 
    
2. `dataGenerator.py`  
   generate classification data from folder structure into a single json file
   partConverter.py: use FreeCAD python API to convert step into brep
   OccQt: generate geometry metadata and dump views into images

3. `dataCollector.py`  
   resize and merge images (opencv2) into numpy.array 
   json meta files into pandas DF

4. `partClassify.py`: TensorFlow model mixed data (images, category data)

5. post processing plot

## Todo

### image resolution + one more view
60X60, so random padding, 
plot_views

do not normalize bbox to cube


MVCNN got running
SPnet:  is also very fast, small in trainable parameter, 30K
pairwise: 

### Rotated view

debug

### ModelNet40

## improve accuracy

### Tuner

https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html

#### Conv2D(filters) choose filters number

[](https://stackoverflow.com/questions/52447345/choosing-conv2d-filters-value-to-start-off-with/52448264)
16 or 32 is fine.  for input image size of 200X200

The filters in the first few layers are usually less abstract and typically emulates edge detectors, blob detectors etc. You generally don't want too many filters applied to the input layer as there is only so much information extractable from the raw input layer. Most of the filters will be redundant if you add too many. You can check this by pruning (decrease number of filters until your performance metrics degrade)


Hyperparameter    |Value             |Best Value So Far 
units             |16                |16                
filters_0         |16                |24                
filters_1         |48                |48                
filters_2         |128               |96                
dropout           |0                 |0.4               
tuner/epochs      |2                 |2                 
tuner/initial_e...|0                 |0                 
tuner/bracket     |3                 |3                 
tuner/round       |0                 |0     


Optuna 是一个特别为机器学习设计的自动超参数优化软件框架。

Introduction to the Keras Tuner
https://www.tensorflow.org/tutorials/keras/keras_tuner


[The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches]
> 2012 the error for AlexNet model using ImageNet data is 16.4%

1. adjust model parameter: like output layer, CNN layer parameters
2. use different image generator!
3. numeric parameters


look at multiple view AI model, for CNN setup

thickness project,  do not scale the view, use orginal aspect ratio
due to different orientation, some part has different contacted images.
OBB. 

64X64 is more than enough, try 32, yes now possible
ModelNet data subset is usable now. 


### Data augmentation to prevent overfitting 

**Data Augmentation** is a method of artificially creating a new dataset for training from the existing training dataset to improve the performance of deep learning neural networks with the amount of data available.

`Model.fit_generator()` is used when either we have a huge dataset to fit into our memory or when data augmentation needs to be applied.

https://my.oschina.net/u/4067628/blog/4767106
+ flip,  must have
在 OpenCV 中 flip 函数的参数有 1 水平翻转、0 垂直翻转、-1 水平垂直翻转三种。 
Tensorflow has experimental data augmentation layer for flip and rotate

+ shift: padding

+ rotate,  not necessary for oriented shape
```py
cv2.getRotationMatrix2D((center_x, center_y) , angle, 1.0)newimg = cv2.warpAffine(img, rotMat, (width, height))
newimg = cv2.warpAffine(img, rotMat, (width, height))
```
+ noise:


https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

https://analyticsindiamag.com/multi-label-image-classification-with-tensorflow-keras/
```py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dataGen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

train_generator = train_dataGen.flow_from_dataframe(
                                        dataframe = training_set,
                                        directory="",x_col="Images",
                                        y_col="New_class",
                                        class_mode="categorical",
                                        target_size=(128,128),
                                        batch_size=32)
```

## Preprocessor

### Geometrical properties: generated by occQt/OCCT, FreeCAD
name of the part
solid count
characteristics length, 2 length ratios. OBB  boundbox
volume/boundbox_volume
volume, area, perimeter,

### Multi-view render occQt/VTK/Blender


views are orth, along OBB


训练网络的时候用的是voxel grids格式的数据，shapeNet提供了32×32×32的grid数据以及grid数据相应渲染的结果，链接：（Index of /data2），里面grid数据是用.binvox格式存储的，python的读取示例（dimatura/binvox-rw-py），如果想要将mesh数据体素化，可以用 mesh-voxelization工具(FairyPig/mesh-voxelization)。

### Xray/thickness map: occProjector/vtk

VTK raycast
vtkMassProperties Class Reference:  `Volume` and `SurfaceArea`


## comparison study

### compressed image, vs not compressed

bit compression, 


### different image resolution, 

quick enough by threading

image pixel is not fixed, why?

scale to aspect ratio 1:1

### using geometry meta data only or images only


