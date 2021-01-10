#!/usr/bin/python3
# -*- coding: utf-8 -*-

#import internal and third party modules
import logging
import sys
import time
import os
import argparse

#import own modules
import lib_IO_hdf5
import model_keras
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))

#Information
__author__ = "Tobias Grundmann, Adrian Schneuwly, Johannes Oswald"
__copyright__ = "Copyright 2016, 3D Vision, ETH Zurich, CVP Group"
__credits__ = ["Martin Oswald", "Pablo Speciale"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Finished"

#set logging level DEBUG and output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="Run voxnex with keras")

    parser.add_argument("dataset", nargs='?', default=script_dir + os.path.sep + "data/modelnet10.hdf5",
                        help="dataset for training in hdf5 format")

    parser.add_argument("-b", "--batch", metavar="size", type=int, default=12,
                        dest="batch_size", help="batch size")

    parser.add_argument("-e", "--epochs", metavar="epochs", type=int, default=80,
                        dest="nb_epoch", help="number of epoches")

    parser.add_argument("-s", "--shuffle_off", action="store_false",
                        dest="shuffle", help="shuffle the data after each epoch")

    parser.add_argument("-r", "--rotation", action="store_true",
                        dest="has_rot", help="decides if the code chould search for rotations, requires an info file")

    parser.add_argument("-v", "--validate", metavar="ratio", type=float, default=0.12,
                        dest="valid_split", help="ratio of training data that should be used for validation, float in range (0,1)")

    parser.add_argument("-c", "--continue", metavar="weights",
                        dest="weights_file", help="continue training, start from given weights file")

#     parser.add_argument("-m", "--mode",default="train", choices=["train", "valid", "test"],
#                         help="set to be returned")

    parser.add_argument("-i", "--interactive_fail",action="store_true",
                        dest="interactive_fail", help="on training fail interactive python console will be launched")

    parser.add_argument("-V", "--verbosity",type=int, default=2, choices=[0, 1, 2],
                        dest="verbosity", help="verbosity setting for training {0,1,2}")

    parser.add_argument("-E", "--evaluate", metavar="eval_weights",
                        dest="eval_weights_file", help="evaluate weights file, start from given weights file")

    # parse arguments
    args = parser.parse_args()

    #quit script in case the file of the dataset is not found
    if not os.path.exists(args.dataset):
        logging.error("[!] File does not exist '{0}'".format(args.dataset))
        sys.exit(-1)

    # start recording time
    tic = time.time()

    # if something crashes, start interpreter shell
    try:

        logging.debug("Using Conversion Method to load HDF5 Data")
        #load dataset from hdf5 file, take batchsize, shuffle option, rotation option and validation split from parser
        loader = lib_IO_hdf5.Loader_hdf5_Convert_Np(args.dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.shuffle,
                                                    has_rot=args.has_rot,
                                                    valid_split=args.valid_split)

        # find dataset name
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

        # create the model
        voxnet = model_keras.model_vt(nb_classes=loader.return_nb_classes(), dataset_name=dataset_name)

        #in case of pretrained weights load them into the model
        if args.eval_weights_file is not None:
            voxnet.load_weights(args.eval_weights_file)

        # train model
        elif args.weights_file is None:
            #if no weights are given train model from scratch using the options returned by the loader
            voxnet.fit(generator=loader.train_generator(),
                       samples_per_epoch=loader.return_num_train_samples(),
                       nb_epoch=args.nb_epoch,
                       valid_generator=loader.valid_generator(),
                       nb_valid_samples=loader.return_num_valid_samples(),
                       verbosity=args.verbosity,
                       )

        else:
            #check if weights file can be found
            if not os.path.exists(args.weights_file):
                logging.error("[!] File does not exist '{0}'".format(args.weights_file))
                sys.exit(-2)

            #if pretrained weights file exits, continue training of weights
            voxnet.continue_fit(weights_file=args.weights_file,
                                generator=loader.train_generator(),
                                samples_per_epoch=loader.return_num_train_samples(),
                                nb_epoch=args.nb_epoch,
                                valid_generator=loader.valid_generator(),
                                nb_valid_samples=loader.return_num_valid_samples())

        # evaluate training on test dataset, !independet of validation dataset!
        voxnet.evaluate(evaluation_generator=loader.evaluate_generator(),
                        num_eval_samples=loader.return_num_evaluation_samples())

    except:
        #in case of failuer while training and with the interactive option given, start interactive python shell,
        # else quit program
        logging.error("Error: Training failed")
        if args.interactive_fail == True:
            logging.debug("Starting Interactive Python Console")
            import code

            if sys.platform.startswith("linux"):
                try:
                    # Note: How to install python3 module readline
                    # sudo apt-get install python3-pip libncurses5-dev
                    # sudo -H pip3 install readline
                    # Note by Tobi: euryale running in virtualenv with python2 currently
                    import readline
                except ImportError:
                    pass

            vars_ = globals().copy()
            vars_.update(locals())
            shell = code.InteractiveConsole(vars_)
            shell.interact()
        else:
            logging.debug("Shutting Program down")
            sys.exit(-2)

    #return the time that has lapsed for the training
    tictoc = time.time() - tic
    print("the run_keras with Conversion to Numpy took {0} seconds".format(tictoc))

    tic = time.time()

if __name__ == "__main__":
    main()
