"""

"""

_is_debug = False
_using_saved_model = True  # 
_using_mixed_input = True

# before import tensorflow
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# import the necessary packages
import tensorflow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

if _is_debug:
    tf.debugging.experimental.enable_dump_debug_info(
        "/tmp/tfdbg2_logdir",
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1)
    # after running the model: run `tensorboard --logdir /tmp/tfdbg2_logdir`
    # NO_TENSOR, CURT_HEALTH, CONCISE_HEALTH, FULL_HEALTH


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Concatenate, Flatten

#from tensorflow.feature_column import categorical_column_with_identity
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import argparse
import locale
import os
import json

from input_parameters import *
from stratify import my_split

#import datasets
#git clone https://github.com/emanhamed/Houses-dataset
from tf_model import create_mlp, create_cnn

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", type=str, required=True,
#    help="path to input metadata and image data")
#args = vars(ap.parse_args())

##################################
# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading classification data")
df = pd.read_json(processed_metadata_filepath)

images = np.load(processed_imagedata_filename)  # pickle array of object type: allow_pickle=True
print("[INFO] loaded images ndarray shape", images.shape)

# there is no need for reshape, np.stack(imagelist)
#images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

if datasetName == "Thingi10K":
    CATEGORY_LABEL="Category"  # "Category" is too coarse to classify
    SUBCATEGORY_LABEL="Subcategory"  # "Subcategory": "rc-vehicles",
    BBOX_LABEL="bbox"
    categories = pd.Series(df["Category"], dtype="category")

    FEATURES_INT = []

else:  # FreeCADLib
    CATEGORY_LABEL="category"
    BBOX_LABEL="obb"
    categories = pd.Series(df["category"], dtype="category")
    FEATURES_INT = ["solidCount", "faceCount"]

####################################
# generate a few new columns based on oriented boundbox, a few ratios
#print(df.head())
bbox = np.stack(df[BBOX_LABEL].to_numpy())

obb = np.zeros((bbox.shape[0], 3))
if bbox.shape[1] == 6:
    obb[:, 0] = bbox[:, 3] - bbox[:, 0]
    obb[:, 1] = bbox[:, 4] - bbox[:, 1]
    obb[:, 2] = bbox[:, 5] - bbox[:, 2]
if bbox.shape[1] == 3:
    obb = bbox

assert((obb > 0).all())

## debug print out obb sorting
#print(obb.shape, obb.dtype, obb[1])
#print(obb[0])
obb.sort(axis=1)  # return None, inplace sort,  ascending order
#print(obb[0])
df["bbox_max_length"] = obb[:,2]
bbox_volume = np.prod(obb)
df["volume_bbox_ratio"] = df["volume"]/bbox_volume
#print(df["bbox_max_length"])

df["obb_ratio_1"]  = obb[:,0] / df["bbox_max_length"] 
df["obb_ratio_2"]  = obb[:,1] / df["bbox_max_length"] 

# axis, symmetric is an important feature

#####################################
imageShape = images.shape

# scaled to value between 0~1.0
FEATURES_RATIO = ["volume_bbox_ratio", "obb_ratio_1", "obb_ratio_2"]

FLOAT_SCALED = ["area_linear", "volume_linear"]
# linear comparable
df["area_linear"] = df["area"]**0.5
df["volume_linear"] = df["volume"]**0.333333333333

raw_FEATURES = FEATURES_RATIO + FEATURES_INT + FLOAT_SCALED
FEATURES = [c+"_scaled" for c in raw_FEATURES]
scaler = MinMaxScaler().fit(df[raw_FEATURES])  ## input must be 2D array
scaled_feature_array = scaler.fit_transform(df[raw_FEATURES])  # fine here, scaled to [0, 1]

for i, c in enumerate(raw_FEATURES):  # feature_columns
    df[c+"_scaled"] = scaled_feature_array[:, i]  

### bug: all scaled data are nan, why?

# some cat has very few samples, make sure they are in train set

##################################

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

################### label encoding ##########
LABEL = "category_classified"
using_feature_column = False
if using_feature_column:
    # todo: feature_column is not defined
    classes = feature_column.categorical_column_with_vocabulary_list(
          CATEGORY_LABEL, df[CATEGORY_LABEL].unique())

    df[LABEL] = feature_column.indicator_column(classes)
    demo(df[LABEL])
else:
    # inverse_transform() to get string back
    catEncoder = LabelEncoder()
    cat = catEncoder.fit_transform(df[CATEGORY_LABEL])
    print("label category encoded as integer: ", type(cat), cat.shape, cat.dtype)  #ndarray of integer
    df[LABEL] = cat


###################################
if _is_debug: 
    print(list(df.columns))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] split data...")


#split = train_test_split(df, images, test_size=0.25, random_state=42)
#(trainDataset, testDataset, trainImagesX, testImagesX) = split

(trainDataset, testDataset, trainImagesX, testImagesX) = my_split(df, images, LABEL)

trainY = pd.DataFrame(trainDataset, columns = [LABEL])
testY = pd.DataFrame(testDataset, columns = [LABEL])

if _is_debug:
    print("trainDataset ", trainDataset.shape, trainDataset)
    print("testDataset ", testDataset.shape, testDataset)

    print("trainY before encoding ", trainY.shape)
    print("testY before encoding ", testY.shape)

#cat_one_hot = tf.one_hot(cat, np.max(cat))  # return a Tensor
onehotencoder = OneHotEncoder()  # need 2D array instead of 1D
#trainY = onehotencoder.fit_transform(trainY.reshape(trainY.shape[0],1)).toarray()
trainY = onehotencoder.fit_transform(trainY).toarray()
testY = onehotencoder.fit_transform(testY).toarray()
total_classes = trainY.shape[1]
# why the array `testY` 's shape is [itemsize, 3]
# WARN: group size is too small to split, skip this group

if _is_debug:
    print("trainY with one-hot encoding ", trainY.shape, trainY)
    print("testY with one-hot encoding ", testY.shape, testY)

from tensorflow.keras.utils import to_categorical
#trainY = to_categorical(trainY)  #return None
#testY = to_categorical(testY)

# no category columns, 
trainAttrX = pd.DataFrame(trainDataset, columns = FEATURES)
testAttrX = pd.DataFrame(testDataset, columns = FEATURES)

##########################################
if _using_saved_model and os.path.exists(saved_model_file):
    print("[INFO] load previously saved model file: ", saved_model_file)
    model = tensorflow.keras.models.load_model(saved_model_file)
else:
    print("[INFO] image shape, and images shape", result_shape, imageShape)
    # from (16, 48, 1) to (1920, 16, 48)
    #assert result_shape[0]  == imageShape[0]
    cnn = create_cnn(*result_shape,  regress=False)  # single sample image input here

    # create the MLP and CNN  models, number of columns
    if _using_mixed_input:
        print("[INFO] build multiple parameters data frame shape", trainAttrX.shape)
        mlp = create_mlp(trainAttrX.shape[1], regress=False)

        # create the input to our final set of layers as the *output* of both
        # the MLP and CNN
        # Flatten()()
        combinedInput = Concatenate(axis=1)([mlp.output, cnn.output])

        # our final FC layer head will have two dense layers, 
        # the final one being our regression head
        x = Dense(4, activation="relu")(combinedInput)  # todo: how to decide the first param? 
        x = Dense(total_classes, activation="softmax")(x) 
        # https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
        # "softmax", "sigmoid" make no diff, is needed for multiple label classification

        # our final  will accept categorical/numerical data on the MLP
        # input and images on the CNN input, outputting a single value (the
        # predicted price of the house)
        model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    else:
        model = cnn

    #########################################
    opt = Adam(lr=1e-5, decay=1e-5 / 200)  # lr: learning rate from 1e-3
    # the loss functions depends on the problem itself, for multiple classification 
    model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=['accuracy'])
    # sparse_categorical_crossentropy

# init the weight values
# train the model
print("[INFO] training part recognition...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX], testY),  # test does not have all classes
    #validation_data=([trainAttrX, trainImagesX], trainY),  #tmp bypass error
    epochs=25, batch_size=20)


#####################################
# save the model and carry on model fit in a second run
# https://www.tensorflow.org/guide/keras/save_and_serialize
model.save(saved_model_file)

########################################
# make predictions on the testing data
preds = model.predict([testAttrX, testImagesX])

# compute the difference between the *predicted*  and the *actual*  
# # then compute the percentage difference 

# 
diff = preds - testY
percentDiff = np.sum(np.sum(np.abs(diff)) / np.sum(testY)) * 100
absPercentDiff = np.abs(percentDiff)
print("[INFO] validate prediction by test data, error percentage is: ", absPercentDiff)

#########################################

total_parameters = 0
for variable in tf.compat.v1.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print(variable, variable_parameters)
    total_parameters += variable_parameters
print("total trainable parameter count = ", total_parameters)