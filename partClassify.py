"""
https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
#
https://analyticsindiamag.com/multi-label-image-classification-with-tensorflow-keras/

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



"""

# before import tensorflow
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# import the necessary packages
import tensorflow
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # WARN =30



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

from fclib_parameters import *
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

df = pd.read_json(processed_metadata_filename)
categories = pd.Series(df["category"], dtype="category")

# load the house images and then scale the pixel intensities to the range [0, 1]
images = np.load(processed_imagedata_filename)  # 
print("loaded images ndarray shape",images.shape)
images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

####################################
# generate a few new columns based on "obb", a few ratios
#print(df.head())
obb = np.stack(df["obb"].to_numpy())
#print(obb.shape, obb.dtype, obb[1])

df["bbox_max_length"] = np.max(obb, axis=1)
bbox_volume = np.prod(obb)
df["volume_bbox_ratio"] = df["volume"]/bbox_volume
#print(df["bbox_max_length"])

## TODO: double check
obb.sort(axis=1)  # return None, inplace sort,  ascending order
df["obb_ratio_1"]  = obb[:,0] / df["bbox_max_length"] 
df["obb_ratio_2"]  = obb[:,1] / df["bbox_max_length"] 

# axis, symmetric is an important feature

#####################################
imageShape = images.shape

# scaled to value between 0~1.0
FEATURES_SCALED = ["volume_bbox_ratio", "obb_ratio_1", "obb_ratio_2"]
FEATURES_INT = ["solidCount", "faceCount"]

FLOAT_SCALED = ["area_linear", "volume_linear"]
# linear comparable
df["area_linear"] = df["area"]**0.5
df["volume_linear"] = df["volume"]**0.333333333333

raw_FEATURES = FEATURES_SCALED + FEATURES_INT + FLOAT_SCALED
FEATURES = [c+"_scaled" for c in raw_FEATURES]
scaler = MinMaxScaler().fit(df[raw_FEATURES])
feature_columns = scaler.fit_transform(df[raw_FEATURES])

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
    classes = feature_column.categorical_column_with_vocabulary_list(
          'category', df["category"].unique())

    df[LABEL] = feature_column.indicator_column(classes)
    demo(df[LABEL])
else:
    # inverse_transform() to get string back
    catEncoder = LabelEncoder()
    cat = catEncoder.fit_transform(df["category"])
    print("label category encoded as integer: ", type(cat), cat.shape, cat.dtype)  #ndarray of integer
    df[LABEL] = cat



###################################
print(list(df.columns))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] split data...")
#feature_columns[LABEL] = df[LABEL]

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

# no category columns
trainAttrX = pd.DataFrame(trainDataset, columns = FEATURES)
testAttrX = pd.DataFrame(testDataset, columns = FEATURES)

##########################################
# create the MLP and CNN  models
mlp = create_mlp(trainAttrX.shape[1], regress=False)
print("image shape, and images shape", result_shape, imageShape)
# (16, 48, 1) (1920, 16, 48)

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

"""
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
"""

#########################################