# -*- coding: utf-8 -*-
"""
the main app for part classification
"""


_debug = True
_using_saved_model = True # 

BATCH_SIZE = 50  # if dataset is small, make this bigger
EPOCH_COUNT = 50
INIT_LEARN_RATE = 1e-4  # batch_normalization needs a slightly bigger learning rate
RESTART_LR = INIT_LEARN_RATE * 0.1

import numpy as np
import pandas as pd
import argparse
import locale
import os
import json
import tempfile


from global_config import dataset_name,  channel_count, thickness_channel, depthmap_channel,  \
    usingOnlyThicknessChannel, usingOnlyDepthmapChannel, usingMixedInputModel, usingKerasTuner, usingMaxViewPooling
from input_parameters import view_count, model_input_shape, processed_metadata_filepath, processed_imagedata_filepath, saved_model_filepath
from stratify import my_split

# before import tensorflow
import logging
#logging.getLogger("tensorflow").setLevel(logging.ERROR)

# import the necessary packages
import tensorflow
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
if _debug:
    tf.get_logger().setLevel('DEBUG')
else:
    tf.get_logger().setLevel('ERROR')

# Set CPU as available physical device, to suppress GPU not found error
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

if _debug:
    tf_debug_dir = tempfile.gettempdir() + os.path.sep + dataset_name + "_logdir"
    tf.debugging.experimental.enable_dump_debug_info(tf_debug_dir,
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1)
    # NO_TENSOR, CURT_HEALTH, CONCISE_HEALTH, FULL_HEALTH
    print("    # after running the model, run:     tensorboard --logdir ", tf_debug_dir)



from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD


#from tensorflow.feature_column import categorical_column_with_identity
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


from DTVmodel import DTVModel

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", type=str, required=True,
#    help="path to input metadata and image data")
#args = vars(ap.parse_args())

##################################
print("[INFO] loading classification data in metadata file")
df = pd.read_json(processed_metadata_filepath)

images = np.load(processed_imagedata_filepath)  # pickle array of object type: allow_pickle=True
print("[INFO] loaded images ndarray shape from file", images.shape, images.dtype)

if len(images.shape) == 5:
    if usingOnlyThicknessChannel:
        images = images[:, :view_count, :, :, thickness_channel]  # choose only the thickness channel
    elif usingOnlyDepthmapChannel:
        images = images[:, :view_count, :, :, depthmap_channel]  # choose only the thickness channel
    else:
        images = images[:, :view_count, :, :, :channel_count]  # choose the depth and thickness channels


if images.shape[-1] > 3  and len(model_input_shape) > len(images.shape)-1  and model_input_shape[-1] == 1:  
    # if single channel does not have its dim
    new_shape = [images.shape[0]] + list(model_input_shape)
    images = np.reshape(images, new_shape)
print("[INFO] loaded images ndarray shape from file", images.shape)

if dataset_name == "Thingi10K":
    CATEGORY_LABEL="Category"  # "Category" is too coarse to classify
    SUBCATEGORY_LABEL="Subcategory"  # "Subcategory": "rc-vehicles",
    BBOX_LABEL="bbox"
    categories = pd.Series(df["Category"], dtype="category")

    FEATURES_INT = []

else:  # FreeCADLib  or ModelNet
    CATEGORY_LABEL="category"
    BBOX_LABEL="obb"
    categories = pd.Series(df["category"], dtype="category")
    if "solidCount" in df.columns and  "faceCount" in df.columns:
        FEATURES_INT = ["solidCount", "faceCount"]
    else:
        FEATURES_INT = []  # mesh input has no such fields

####################################
# generate a few new columns based on oriented boundbox, a few ratios
#print(df.head())
try:
    bbox = np.stack(df[BBOX_LABEL].to_numpy())
except:
    bbox = np.stack(df['bbox'].to_numpy())

obb = np.zeros((bbox.shape[0], 3))
if bbox.shape[1] == 6:
    obb[:, 0] = bbox[:, 3] - bbox[:, 0]
    obb[:, 1] = bbox[:, 4] - bbox[:, 1]
    obb[:, 2] = bbox[:, 5] - bbox[:, 2]
if bbox.shape[1] == 3:
    obb = bbox

# there is obb length are zeros, lead to bbox_volume to zero.
#print(obb<=0)
assert((obb > 0).all())


## debug print out obb sorting
#print(obb.shape, obb.dtype, obb[1])
#print(obb[0])
obb.sort(axis=1)  # return None, inplace sort,  ascending order
#print(obb[0]),   # obb max is too big, why?
df["bbox_max_length"] = obb[:,2]
bbox_volume = np.prod(obb, axis=1)
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
    print("label category encoded as integer: ", type(cat), cat.shape, cat.dtype)  # ndarray of integer
    df[LABEL] = cat


###################################
if _debug: 
    print(list(df.columns))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] split data...")


#split = train_test_split(df, images, test_size=0.25, random_state=42)
#(trainDataset, testDataset, trainImagesX, testImagesX) = split

if dataset_name == "ModelNet":
    train_folder_name="train"
else:
    train_folder_name=None
(trainDataset, testDataset, trainImagesX, testImagesX) = my_split(df, images, LABEL, train_folder_name=train_folder_name)

trainY = pd.DataFrame(trainDataset, columns = [LABEL])
testY = pd.DataFrame(testDataset, columns = [LABEL])

if _debug:
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

print("[INFO] trainY shape by one-hot encoding ", trainY.shape)
if _debug:
    print("testY with one-hot encoding ", testY.shape, testY)


#trainY = to_categorical(trainY)  #return None
#testY = to_categorical(testY)

# no category columns, 
trainAttrX = pd.DataFrame(trainDataset, columns = FEATURES)
testAttrX = pd.DataFrame(testDataset, columns = FEATURES)

##########################################
## auto checkpoint save?

# EarlyStopping for prevent overfitting
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2)

# keras.callbacks.ModelCheckpoint
checkpoint_filepath = tempfile.tempdir + '/tf_checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

import signal
def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    # if save model, it that a reloadable model?  
    exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)

##########################################
print("[INFO] model input image shape, and images shape", model_input_shape, imageShape)
model_settings = { "total_classes": total_classes, "usingMixedInputs": usingMixedInputModel,
                    "usingMaxViewPooling": usingMaxViewPooling,
                    "image_width": model_input_shape[-3], "regress": False}

# LearningRate = LearningRate * 1/(1 + decay * epoch)
#opt = Adam(learning_rate=INIT_LEARN_RATE, beta_1=0.7, decay=50/EPOCH_COUNT)
opt = Adam(lr=INIT_LEARN_RATE, beta_1=0.7, decay=1e-5 )  #  1e-5 / 200

if usingKerasTuner:
    print("[INFO] use keras tuner")
    def build_model(hp):
        model = TDModel(model_settings, hp).create_model(im_shape = model_input_shape, mlp_shape = trainAttrX.shape)
        model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=['accuracy'])
        return model

    import kerastuner as kt

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,  # can that based on a pretrained model?
        hyperband_iterations=4)

    # TODO: error: ValueError: Layer model_1 expects 2 input(s), but it received 3 input tensors.
    tuner.search([trainAttrX, trainImagesX], trainY,
                validation_data=([testAttrX, testImagesX], testY),
                epochs=30,  # this is not enough for this kind of model
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    best_model = tuner.get_best_models(1)[0]
    best_model.save(saved_model_filepath)
else:

    if _using_saved_model and os.path.exists(saved_model_filepath):
        print("[INFO] load previously saved model file: ", saved_model_filepath)

        model = tensorflow.keras.models.load_model(saved_model_filepath)
        model_loaded = True
    else:

        model = TDModel(model_settings).create_model(im_shape = model_input_shape, mlp_shape = trainAttrX.shape)

        # the loss functions depends on the problem itself, for multiple classification 
        if True:
            model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=['accuracy'])
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,  metrics=['accuracy'])
            # sparse_categorical_crossentropy is for large class count, also needs to change label encoder?

    #########################################

    # is that possible to set some model parameters after load, YES
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
    from tensorflow.keras import backend as K
    print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())
    K.set_value(model.optimizer.learning_rate, RESTART_LR)


    # init the weight values
    print("[INFO] training part recognition...")
    callbacks=[model_checkpoint_callback]

    if usingMixedInputModel:
        history = model.fit(
            [trainAttrX, trainImagesX], trainY,
            validation_data=([testAttrX, testImagesX], testY),  # test does not have all classes
            epochs=EPOCH_COUNT, batch_size=BATCH_SIZE)
    else:
        history = model.fit(
            trainImagesX, trainY,
            validation_data=(testImagesX, testY),  # test does not have all classes
            epochs=EPOCH_COUNT, batch_size=BATCH_SIZE)

    #####################################
    # save the model and carry on model fit in a second run
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    model.save(saved_model_filepath, save_format='h5')
    # save_format='h5' is fine if tf.debugging is enabled, for some dtype cause error for save_format = 'tf'.

    # convert the history.history dict to a pandas DataFrame:
    import pandas as pd 
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = saved_model_filepath + '.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    ########################################
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")

    # make predictions on the testing data
    if usingMixedInputModel:
        preds = model.predict([testAttrX, testImagesX])
        results = model.evaluate([testAttrX, testImagesX], testY, batch_size=128)
    else:
        preds = model.predict(testImagesX)
        results = model.evaluate(testImagesX, testY, batch_size=128)

    print("[INFO] test loss, test acc:", results)

    # compute the difference between the *predicted*  and the *actual*  
    # # then compute the percentage difference 

    # this is twice of loss evaluation
    # diff = preds - testY
    # percentDiff = np.sum(np.sum(np.abs(diff)) / np.sum(testY)) * 100
    # absPercentDiff = np.abs(percentDiff)
    # print("[INFO] validate prediction by test data, error percentage is: ", absPercentDiff)

    #########################################
