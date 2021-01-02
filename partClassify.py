"""

"""

# before import tensorflow
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# import the necessary packages
import tensorflow
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # WARN =30

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
print("loaded images ndarray shape", images.shape)
print("loaded images ndarray shape", images[0].shape)
# there is no need for reshape, np.stack(imagelist)
#images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

if datasetName == "Thingi10K":
    CATEGORY_LABEL="Category"  
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
if obb.shape[1] == 6:
    obb[:, 0] = bbox[:, 3] - bbox[:, 0]
    obb[:, 1] = bbox[:, 4] - bbox[:, 1]
    obb[:, 2] = bbox[:, 5] - bbox[:, 2]
else:  # shape[1] == 3
    obb = bbox

assert((obb > 0).all())
print(obb.shape, obb.dtype, obb[1])

## TODO: double check
print(obb[0])
obb.sort(axis=1)  # return None, inplace sort,  ascending order
print(obb[0])
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
print(list(df.columns))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] split data...")


#split = train_test_split(df, images, test_size=0.25, random_state=42)
#(trainDataset, testDataset, trainImagesX, testImagesX) = split

(trainDataset, testDataset, trainImagesX, testImagesX) = my_split(df, images, LABEL)

trainY = pd.DataFrame(trainDataset, columns = [LABEL])
testY = pd.DataFrame(testDataset, columns = [LABEL])

print("trainDataset ", trainDataset.shape, trainDataset)
print("testDataset ", testDataset.shape, testDataset)

print("trainY before encoding ", trainY.shape)
print("testY before encoding ", testY.shape)

#cat_one_hot = tf.one_hot(cat, np.max(cat))  # return a Tensor
onehotencoder = OneHotEncoder()  # need 2D array instead of 1
#trainY = onehotencoder.fit_transform(trainY.reshape(trainY.shape[0],1)).toarray()
trainY = onehotencoder.fit_transform(trainY).toarray()
testY = onehotencoder.fit_transform(testY).toarray()
total_classes = trainY.shape[1]

print("trainY with one-hot encoding ", trainY.shape, trainY)
print("testY with one-hot encoding ", testY.shape, testY)

from tensorflow.keras.utils import to_categorical
#trainY = to_categorical(trainY)  #return None
#testY = to_categorical(testY)

# no category columns, 
trainAttrX = pd.DataFrame(trainDataset, columns = FEATURES)
testAttrX = pd.DataFrame(testDataset, columns = FEATURES)

##########################################
# create the MLP and CNN  models, number of columns
print("multiple parameters data frame shape", trainAttrX.shape)
mlp = create_mlp(trainAttrX.shape[1], regress=False)

print("image shape, and images shape", result_shape, imageShape)
# from (16, 48, 1) to (1920, 16, 48)
#assert result_shape[0]  == imageShape[0]
cnn = create_cnn(*result_shape,  regress=False)  # single sample image input here

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

# todo:  simplify image model

#########################################
opt = Adam(lr=1e-6, decay=1e-5 / 200)  # from 1e-3 to
# the loss functions depends on the problem itself, for multiple classification 
model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=['accuracy'])
# sparse_categorical_crossentropy

# train the
print("[INFO] training part recognition...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX], testY),  # test does not have all classes
    #validation_data=([trainAttrX, trainImagesX], trainY),  #tmp bypass error
    epochs=5, batch_size=20)

# make predictions on the testing data
print("[INFO] predicting shape class")
preds = model.predict([testAttrX, testImagesX])

########################################
# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference

# todo: decode back then calc
diff = preds - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)



#########################################
