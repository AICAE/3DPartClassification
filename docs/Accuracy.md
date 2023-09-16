## improve accuracy

### TensorFlow Tuner

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
4. LDF

look at multiple view AI model, for CNN setup

thickness project,  do not scale the view, use orginal aspect ratio
due to different orientation, some part has different contacted images.
OBB. 

64X64 is more than enough, try 32, yes now possible
ModelNet data subset is usable now.

### invalid dataset
http://edge.cs.drexel.edu/repository/

(https://www.researchgate.net/publication/327272531_A_methodology_for_part_classification_with_supervised_machine_learning)

A methodology for part classification with
supervised machine learning
http://partsclassifier.ge.imati.cnr.it

############################
1508 training set for FreeCAD-lib,  OBBbox,  78%, maybe due to too small the traing set.

diff of Conv2D on grayscale and RGB images

They will be just the same as how you do with a single channel image, except that you will get three matrices instead of one. This is a lecture note about CNN fundamentals, which I think might be helpful for you.

https://www.researchgate.net/post/How-will-channels-RGB-effect-convolutional-neural-network
> in the case of matlab: if you have a colored (RGB) image with 3 channels, and you carry out 2D convolution on it; and if you define filter size as (5x5), matlab automatically creates a filter with 3 channels. so the number of neurons in this filter will be 5x5x3. The output of the filter will be the sum of the dot product from 5x5x3 terms + 1 bias.
###########################################

BATCH_SIZE = 1000  # if dataset is small, make this bigger
EPOCH_COUNT = 30
INIT_LEARN_RATE = 3e-5  # batch_normalization needs a slightly bigger learning rate
RESTART_LR = INIT_LEARN_RATE * 0.1

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
