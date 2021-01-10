#!/usr/bin/python3
# -*- coding: utf-8 -*-

#internal modules
import logging
import datetime
import os
import sys

#adapted to tensorflow 2.x
import tensorflow.keras as keras
print(keras.__version__)

#assert keras.__version__ == "1.0.0", "keras version not supported"
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution3D, MaxPooling3D
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD

#Information
__author__ = "Tobias Grundmann, Adrian Schneuwly, Johannes Oswald"
__copyright__ = "Copyright 2016, 3D Vision, ETH Zurich, CVP Group"
__credits__ = ["Martin Oswald", "Pablo Speciale"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Finished"

#set logging level DEBUG and output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def learningRateSchedule(epoch):
    """

    Args:
        epoch: current epoche, int

    Returns:
        learning rate based on epoche

    Learning Rate Schedule which returns changing learning rates based on the epoche

    """
    if epoch >= 2:
        return 0.0001
    elif epoch >= 5:
        return 0.00005,
    elif epoch >= 15:
        return 0.00001
    else:
        return 0.001

class model_vt (object):
    """

    Reimplementation of the voxnet by dimatura

    """
    def __init__(self, nb_classes, dataset_name):
        """

        Args:
            nb_classes: number of classes the model is going to learn, int
            dataset_name: name of the dataset {modelnet40, modelnet10} just used to save weights every epoche

        initializes voxnet based on keras framework

        layers:
            3D Convolution
            Leaky ReLu
            Dropout
            3d Convolution
            Leaky ReLu
            MaxPool
            Dropout
            Dense
            Dropout
            Dense

        """

        # Stochastic Gradient Decent (SGD) with momentum
        # lr=0.01 for LiDar dataset
        # lr=0.001 for other datasets
        # decay of 0.00016667 approx the same as learning schedule (0:0.001,60000:0.0001,600000:0.00001)
        self._optimizer = SGD(lr=0.01, momentum=0.9, decay=0.00016667, nesterov=False)

        # use callbacks learingrate_schedule as alternative to learning_rate decay
        #   self._lr_schedule = LearningRateScheduler(learningRateSchedule)

        # save weights after every epoche
        self._mdl_checkpoint = ModelCheckpoint("weights/" + dataset_name + "_{epoch:02d}_{acc:.2f}.hdf5",
                                               monitor="acc", verbose=0, save_best_only=False, mode="auto")

        # create directory if necessary
        if not os.path.exists("weights/"):
            os.makedirs("weights/")

        # init model
        self._mdl = Sequential()

        # convolution 1,  TODO:  error here!!!
        self._mdl.add(Convolution3D(input_shape=(1, 32, 32, 32),
                                    nb_filter=32,
                                    kernel_dim1=5,
                                    kernel_dim2=5,
                                    kernel_dim3=5,
                                    init='normal',
                                    border_mode='valid',
                                    subsample=(2, 2, 2),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    ))

        logging.debug("Layer1:Conv3D shape={0}".format(self._mdl.output_shape))
        #Activation Leaky ReLu
        self._mdl.add(Activation(LeakyReLU(alpha=0.1)))

        # dropout 1
        self._mdl.add(Dropout(p=0.3))

        # convolution 2
        self._mdl.add(Convolution3D(nb_filter=32,
                                    kernel_dim1=3,
                                    kernel_dim2=3,
                                    kernel_dim3=3,
                                    init='normal',
                                    border_mode='valid',
                                    subsample=(1, 1, 1),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    ))
        logging.debug("Layer3:Conv3D shape={0}".format(self._mdl.output_shape))

        #Activation Leaky ReLu
        self._mdl.add(Activation(LeakyReLU(alpha=0.1)))

        # max pool 1
        self._mdl.add(MaxPooling3D(pool_size=(2, 2, 2),
                                   strides=None,
                                   border_mode='valid',
                                   dim_ordering='th'))
        logging.debug("Layer4:MaxPool3D shape={0}".format(self._mdl.output_shape))

        # dropout 2
        self._mdl.add(Dropout(p=0.4))

        # dense 1 (fully connected layer)
        self._mdl.add(Flatten())
        logging.debug("Layer5:Flatten shape={0}".format(self._mdl.output_shape))

        self._mdl.add(Dense(output_dim=128,
                            init='normal',
                            activation='linear',
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            ))
        logging.debug("Layer6:Dense shape={0}".format(self._mdl.output_shape))

        # dropout 3
        self._mdl.add(Dropout(p=0.5))

        # dense 2 (fully connected layer)
        self._mdl.add(Dense(output_dim=nb_classes,
                            init='normal',
                            activation='linear',
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            ))
        logging.debug("Layer8:Dense shape={0}".format(self._mdl.output_shape))

        #Activation Softmax
        self._mdl.add(Activation("softmax"))

        # compile model
        self._mdl.compile(loss='categorical_crossentropy', optimizer=self._optimizer, metrics=["accuracy"])
        logging.info("Model compiled!")

    def fit(self, generator, samples_per_epoch,
            nb_epoch, valid_generator, nb_valid_samples, verbosity):
        """

        Args:
            generator: training sample generator from loader.train_generator
            samples_per_epoch: number of train sample per epoche from loader.return_train_samples
            nb_epoch: number of epochs to repeat traininf on full set
            valid_generator: validation sample generator from loader.valid_generator or NONE else
            nb_valid_samples: number of validation samples per epoche from loader.return_valid_samples
            verbosity: 0 (no output), 1 (full output), 2 (output only after epoche)

        """
        logging.info("Start training")
        self._mdl.fit_generator(generator=generator,
                                samples_per_epoch=samples_per_epoch,
                                nb_epoch=nb_epoch,
                                verbose=verbosity,
                                callbacks=[ #self._lr_schedule,
                                        self._mdl_checkpoint,],
                                validation_data=valid_generator,
                                nb_val_samples=nb_valid_samples,
                                )

        time_now = datetime.datetime.now()
        time_now = "_{0}_{1}_{2}_{3}_{4}_{5}".format(time_now.year, time_now.month, time_now.day,
                                                     time_now.hour, time_now.minute, time_now.second)
        logging.info("save model Voxnet weights as weights_{0}.h5".format(time_now))
        self._mdl.save_weights("weights_{0}.h5".format(time_now), False)

    def continue_fit(self, weights_file, generator, samples_per_epoch,
                     nb_epoch, valid_generator, nb_valid_samples, verbosity):
        """

        Args:
            weights_file: filename and adress of weights file .hdf5
            generator: training sample generator from loader.train_generator
            samples_per_epoch: number of train sample per epoche from loader.return_train_samples
            nb_epoch: number of epochs to repeat traininf on full set
            valid_generator: validation sample generator from loader.valid_generator or NONE else
            nb_valid_samples: number of validation samples per epoche from loader.return_valid_samples
            verbosity: 0 (no output), 1 (full output), 2 (output only after epoche)

        """
        self.load_weights(weights_file)
        self._mdl.fit_generator(generator=generator,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=verbosity,
                            callbacks=[ #self._lr_schedule,
                                    self._mdl_checkpoint,],
                            validation_data=valid_generator,
                            nb_val_samples=nb_valid_samples,
                            )

    def evaluate(self, evaluation_generator, num_eval_samples):
        """

        Args:
            evaluation_generator: evaluation sample generator from loader.eval_generator
            num_eval_samples: number of train sample per epoche from loader.return_eval_samples

        """
        self._score = self._mdl.evaluate_generator(
            generator=evaluation_generator,
            val_samples=num_eval_samples)
        print("Test score:", self._score)

    def load_weights(self, file):
        """

        Args:
            file: filename and adress of weights file .hdf5

        """
        logging.info("Loading model weights from file '{0}'".format(file))
        self._mdl.load_weights(file)

    def predict(self, X_predict):
        """

        Args:
            X_predict: Features to use to predict labels, numpy ndarray shape [~,1,32,32,32]

        returns:
            Probability for every label

        """
        return self._mdl.predict_proba(X_predict, verbose=0)
