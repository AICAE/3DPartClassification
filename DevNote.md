
### debug tensorflow

https://www.tensorflow.org/tensorboard/debugger_v2
It can discovery `loss: nan` error

modelnet40,  even the image in first epoch run into `loss: nan` error

`tf.debugging.enable_check_numerics`

`tf.get_logger().setLevel('INFO')`


#### thingi10k dataset

`Subcategory` each cat is too smaller,
while `Category`
152/152 [==============================] - 18s 117ms/step - loss: 3.9862 - accuracy: 0.0250 - val_loss: 3.9852 - val_accuracy: 0.0280



### ModelNet10 dataset

3 channel, randon shift `nearest`
in preprocessor may only pickup only the `usingOnlyThicknessChannel`

about 1800 epoches: 
49/49 [==============================] - 2s 47ms/step - loss: 0.9724 - accuracy: 0.6587 - val_loss: 1.9256 - val_accuracy: 0.6152

Epoch 2800
49/49 [==============================] - 2s 35ms/step - loss: 0.8983 - accuracy: 0.6629 - val_loss: 2.0460 - val_accuracy: 0.6173

change learning rate to 1e-3, will not converge
Epoch 277/1250
49/49 [==============================] - 2s 39ms/step - loss: 1.2822 - accuracy: 0.5112 - val_loss: 1.1587 - val_accuracy: 0.5802


change batch to 5 does not help
1925/1925 [==============================] - 6s 3ms/step - loss: 0.7091 - accuracy: 0.7413 - val_loss: 4.5206 - val_accuracy: 0.6399

diff from LIDAR point cloud,  there is thickness, no RGB color


Without mixed input MLP, 84% only  Jan 31
Epoch 800/800
31/31 [==============================] - 20s 643ms/step - loss: 0.5662 - accuracy: 0.8182 - val_loss: 0.4400 - val_accuracy: 0.8394

max pooling Feb 01  
31/31 [==============================] - 20s 641ms/step - loss: 0.4308 - accuracy: 0.8596 - val_loss: 0.3465 - val_accuracy: 0.8838
Epoch 99/100
31/31 [==============================] - 20s 645ms/step - loss: 0.4429 - accuracy: 0.8540 - val_loss: 0.3455 - val_accuracy: 0.8864
Epoch 100/100
31/31 [==============================] - 20s 638ms/step - loss: 0.4337 - accuracy: 0.8603 - val_loss: 0.3445 - val_accuracy: 0.8838


with MLP  Max pooling?

#### ModelNet40


### Conv2D parameter selections


https://github.com/WeiTang114/MVCNN-TensorFlow/blob/master/model.py

https://github.com/Mannix1994/MVCNN-Keras/blob/master/model.py
Inspired by WeiTang114's project MVCNN-TensorFlow.

image pixels  224X224       the first Conv2D filters = 96,  later 256, 384
more dense layers, with big number

https://github.com/DreamIP/mvcnn/tree/master/fpga
training on FPGA?


Epoch 250/250
2/2 [==============================] - 2s 1s/step - loss: 0.8092 - accuracy: 0.7070 - val_loss: 0.9877 - val_accuracy: 0.7016
after 1250 epochs, val_loss does not go down,  acc is 0.72, seems always
about 2000 epochs,  no further drop,  val_loss even grows up. 
2/2 [==============================] - 2s 1s/step - loss: 0.5927 - accuracy: 0.7901 - val_loss: 1.0992 - val_accuracy: 0.7366
[INFO] validate prediction by test data, error percentage is:  70.06794332800224

fix CPU freq?  more than hardware cores, why?     make some core offline first then start the training, 
choose how many cores to use, not working in TF v2

### split of train/test
DF has the subcategory,  path

### thickness grayscale channel,  Conv kernel

is MVCNN binarized image?


### view pooling, seems not quite working, worse than image concat.

can use concat Layer, instead of MaxPooling layer

Epoch 750/750
[==============================] - 2s 861ms/step - loss: 1.7271 - accuracy: 0.3777 - val_loss: 1.7437 - val_accuracy: 0.4012

concat layer is much better 

Epoch 250/250
2/2 [==============================] - 1s 757ms/step - loss: 2.7867 - accuracy: 0.3012 - val_loss: 1.9338 - val_accuracy: 0.3272
Epoch 500/500
2/2 [==============================] - 1s 708ms/step - loss: 1.9160 - accuracy: 0.3470 - val_loss: 1.8687 - val_accuracy: 0.3539


### Mixed input with MLP

Epoch 250/250
2/2 [==============================] - 2s 830ms/step - loss: 5.2023 - accuracy: 0.2216 - val_loss: 2.4837 - val_accuracy: 0.3272

scalar parameters are volume, aspect ratio, for the first 250 epoch, it seem does not make big diff, 


### model comparision
12X227X227X3 very un efficient
3X60X60X2
32X32X32  Conv3D is more time consuming

trainable parameters
DTMVNet: Trainable params: 2,681,738


nview_processed_imagedata  1GB,  
3808  10 groups

Epoch 1/2
31/31 [==============================] - 22s 665ms/step - loss: 0.1646 - accuracy: 0.9438 - val_loss: 0.2593 - val_accuracy: 0.9191
Epoch 2/2
31/31 [==============================] - 20s 647ms/step - loss: 0.1649 - accuracy: 0.9464 - val_loss: 0.2620 - val_accuracy: 0.9164
2021-01-24 08:44:23.120916: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
[INFO] validate prediction by test data, error percentage is:  19.691952307044723