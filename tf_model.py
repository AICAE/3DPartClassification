import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


'''
# not needed, not completed
class ViewPool(tensorflow.keras.layers.Layer):
    # see also the class Pooling2D layer
    def __init__(self, input_views,
                pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs):
        super(ViewPool, self).__init__()
        w_init = tf.random_normal_initializer()
        nviews = len(input_views)
        view_shape = input_views[0].shape
        input_shape = 
        self.w = tf.Variable(
            initial_value=w_init(shape=input_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
'''

class TDModel(object):
    """ Thickness and depth dual channel
    """
    def __init__(self, s):
        self.settings = s
        self.total_classes = s["total_classes"]
        self.OUT_NODE_COUNT = self.total_classes
        self.usingMixedInputs = s["usingMixedInputs"]
        self.usingMaxViewPooling = False
        self.regress=s["regress"]

    def create_mlp(self, dim):
        # define our MLP network
        # if tensorflow.feature_column is used as input, 
        # then the first layer must be DenseFeatures
        model = Sequential()
        model.add(Dense(8, input_dim=dim, activation="relu"))  # todo: why units = 8 ???
        model.add(Dense(self.OUT_NODE_COUNT, activation="relu"))

        # check to see if the regression node should be added
        if self.regress:
            model.add(Dense(1, activation="linear"))

        return model

    def view_pooling(self, view_pool):
        # concat and max_pool
        #output = tf.keras.layers.Concatenate(view_pool, axis=0)
        #return  tf.keras.layers.Maximum(axis=0)(output)
        if self.usingMaxViewPooling:
            return  tf.keras.layers.Maximum()(view_pool)
        else:
            x = Concatenate(axis=1)(view_pool)
            print("concat shape after view pool is ", x.shape)
            return x


    def create_cnn(self, inputShape, filters=(8, 16, 32), data_augmentation=False):
        # inputShape does not include the batch_size, but view_count
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering
        
        #inputShape = ( height, width,  depth) for color image
        #inputShape = ( width, height) for single channel image
        # if images are not concat, there is another dim in front

        # an extra tf.experimental.image preprocessor has DataAugment Layer to flip and rotate image
        #layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        #layers.experimental.preprocessing.Rescaling(1./255)

        # needs a Conv2D kernel to extract gradient feature from thickness, depth image
        #https://github.com/Mannix1994/MVCNN-Keras/blob/master/model.py

        # define the model input
        viewDim = 0
        nviews = inputShape[viewDim]
        hasMultipleChannel = len(inputShape) >=4  and inputShape[-1] > 1
        usingBatchNormal = False
        chanDim = len(inputShape) - 1  # channel_last
        local_feature_count = 32
        inputs = Input(shape=tuple(inputShape))
        #print(Input.shape)

        view_pool = []
        for v in range(nviews):
            # for single view image, the total dim is one less
            # AUGMENTATION => CONV => RELU => BN => POOL

            x0 = inputs[:, v]
            if data_augmentation:
                x = RandomFlip("horizontal_and_vertical")(x0)  # important if object is not aligned
                x = RandomRotation(0.2)(x)
                # random shift/padding, may also help
            else:
                x = x0  # fetch only a single view image

            # loop over the number of filters
            for (i, f) in enumerate(filters):
                # if this is the first CONV layer then set the input appropriately
                x = Conv2D(f, (3, 3), padding="same")(x)  # how about stride?
                x = Activation("relu")(x)
                if usingBatchNormal:
                    x = BatchNormalization(axis=chanDim)(x)  # not needed for binary image?
                x = MaxPooling2D(pool_size=(2, 2))(x)

            # flatten the volume, then FC => RELU => BN => DROPOUT
            x = Flatten()(x)
            x = Dense(local_feature_count)(x)
            x = Activation("relu")(x)
            if usingBatchNormal:
                x = BatchNormalization(axis=chanDim)(x)  #BN has some Dropout's regularization effect
            else:
                x = Dropout(0.5)(x)
            view_pool.append(x)

        # view pooling
        x = self.view_pooling(view_pool)
        # why reduce_mean? in MVCNN?
        x = Flatten()(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        x = Dense(self.OUT_NODE_COUNT)(x)
        x = Activation("relu")(x)

        if not self.usingMixedInputs:
            x = Dense(self.total_classes, activation="softmax")(x)

        # check to see if the regression node should be added
        if self.regress:
            x = Dense(1, activation="linear")(x)

        # construct the CNN
        model = Model(inputs, x)

        return model

    def create_model(self, im_shape, mlp_shape):
        # from (16, 48, 1) to (1920, 16, 48)
        #assert result_shape[0]  == imageShape[0]
        cnn = self.create_cnn(im_shape)  # single sample image input here

        # create the MLP and CNN  models, number of columns
        if self.usingMixedInputs:
            print("[INFO] build multiple parameters data frame shape", mlp_shape)
            mlp = self.create_mlp(mlp_shape[1])

            # create the input to our final set of layers as the *output* of both
            # the MLP and CNN
            combinedInput = Concatenate(axis=1)([mlp.output, cnn.output])

            # our final FC layer head will have two dense layers, 
            # the final one being our regression head
            x = Dense(self.OUT_NODE_COUNT, activation="relu")(combinedInput)  # todo: how to decide the first param? 
            x = Dense(self.total_classes, activation="softmax")(x) 
            # https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
            # "softmax", "sigmoid" make no diff, is needed for multiple label classification

            # our final  will accept categorical/numerical data on the MLP
            # input and images on the CNN input, outputting a single value (the
            # predicted price of the house)
            model = Model(inputs=[mlp.input, cnn.input], outputs=x)
        else:
            model = cnn

        return model