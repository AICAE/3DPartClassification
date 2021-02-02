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



class TDModel(object):
    """ Thickness and depth dual channel
    tuner is also supported, if 
    """
    def __init__(self, s, hp=None):
        self.settings = s
        self.total_classes = s["total_classes"]
        self.usingMixedInputs = s["usingMixedInputs"]
        self.image_width = s["image_width"]
        self.usingMaxViewPooling = s["usingMaxViewPooling"]

        if hp:
            self.mlp_dense_p = 16 # hp.Int('units', min_value = 16, max_value = 32, step = 8)
            self.local_feature_count = 64
            self.cnn_filters=[]
            if self.image_width <=64:
                bf = (16, 32, 64)
            for i,w in enumerate(bf):
                self.cnn_filters.append(hp.Int('filters_' + str(i), w, w*2, step=w//2))
            self.pre_out_dense_p = hp.Int('preout_dense', 128, 512, step=128)
            self.out_dense_p = hp.Int('out_dense', 128, 512, step=128)
            self.cnn_dense_dropout_p = 0.4 # hp.Float('dropout', 0.2, 0.6, step=0.2, default=0.4)
        else:
            # modelnet10 Jan25, 0.91
            # self.mlp_dense_p = 16
            # self.local_feature_count = 64
            # self.cnn_filters=(24, 48, 64)
            # self.pre_out_dense_p = 384 # layer before output Dense layer
            # self.out_dense_p = 256
            # self.cnn_dense_dropout_p = 0.6

            # Jan 21, Acc = 0.92 in 1000 epochs
            self.mlp_dense_p = 16
            self.local_feature_count = 64
            self.cnn_filters=(16, 32, 64)
            self.pre_out_dense_p = 256 # layer before output Dense layer
            self.out_dense_p = 256
            self.cnn_dense_dropout_p = 0.6

        self.usingDataAugmentation = True
        self.usingBatchNormal = True
        self.regress=s["regress"]

    def create_mlp(self, dim):
        # define our MLP network
        # if tensorflow.feature_column is used as input, 
        # then the first layer must be DenseFeatures
        model = Sequential()
        model.add(Dense(self.mlp_dense_p, input_dim=dim, activation="relu"))  # todo: why units = 8 ???
        model.add(Dense(self.out_dense_p, activation="relu"))

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


    def create_cnn(self, inputShape):
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

        chanDim = - 1  # channel_last

        inputs = Input(shape=tuple(inputShape))
        #print(Input.shape)

        view_pool = []
        for v in range(nviews):
            # for single view image, the total dim is one less
            # AUGMENTATION => CONV => RELU => BN => POOL

            x0 = inputs[:, v]
            if self.usingDataAugmentation:
                x = RandomFlip("horizontal_and_vertical")(x0)  # important if object is not aligned
                x = RandomRotation(0.2)(x)
                # random shift/padding, may also help
            else:
                x = x0  # fetch only a single view image

            # loop over the number of filters, from smaller to bigger
            for (i, f) in enumerate(self.cnn_filters):
                # if this is the first CONV layer then set the input appropriately
                x = Conv2D(f, (3, 3), padding="same", activation='relu')(x)
                # input image size is small, use the default strides = (1,1)
                #x = Activation("relu")(x)  # merged into Conv2D
                if self.usingBatchNormal:
                    x = BatchNormalization(axis=chanDim)(x)  # not needed for binary image?
                x = MaxPooling2D(pool_size=(2, 2))(x)

            # flatten the volume, then FC => RELU => BN => DROPOUT
            x = Flatten()(x)
            # x = Dense(local_feature_count)(x)
            # x = Activation("relu")(x)
            # if usingBatchNormal:
            #     x = BatchNormalization(axis=chanDim)(x)  #BN has some Dropout's regularization effect
            # else:
            #     x = Dropout(0.5)(x)
            view_pool.append(x)

        # view pooling
        x = self.view_pooling(view_pool)

        x = Dense(self.pre_out_dense_p, activation='relu')(x)
        #x = Activation("relu")(x)
        x = Dropout(self.cnn_dense_dropout_p)(x)

        x = Dense(self.out_dense_p, activation='relu')(x)
        #x = Activation("relu")(x)
        x = Dropout(self.cnn_dense_dropout_p)(x)

        # apply another FC layer, this one to match the number of classes
        if not self.usingMixedInputs:
            x = Dense(self.out_dense_p, activation="relu")(x)
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
            x = Dense(self.out_dense_p, activation="relu")(combinedInput)  # todo: how to decide the first param? 
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