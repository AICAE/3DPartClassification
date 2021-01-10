#!/usr/bin/python2
# -*- coding: utf-8 -*-

#load thrid party modules
import numpy as np
import random
from scipy.io import loadmat

#load project modules
import model_keras
from config.model_cfg import class_id_to_name_modelnet40

#Information
__author__ = "Tobias Grundmann, Adrian Schneuwly, Johannes Oswald"
__copyright__ = "Copyright 2016, 3D Vision, ETH Zurich, CVP Group"
__credits__ = ["Martin Oswald", "Pablo Speciale"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Finished"


def load_pc(fname):
    """

    this functions loads a .mat file density grid from a tango tablet

    Args:
        fname: filename of density grid, .mat file

    Returns:
        numpy ndarray with density grid data as float32 type

    """
    f = loadmat(fname)
    data = f['data'].astype(np.float32)
    return data

def voxilize(np_pc, rot = None):
    """

    this function converts a tango tablet matrix into a voxnet voxel volume

    Args:
        np_pc: numpy ndarray with density grid data from load_pc
        rot: ability to rotate picture rot times and take rot recognitions

    Returns:
        voxilized version of density grid that is congruent with voxnet size
    """
    # chance to fill voxel
    p = 80

    max_dist = 0.0
    for it in range(0,3):
        # find min max & distance in current direction
        min = np.amin(np_pc[:,it])
        max = np.amax(np_pc[:,it])
        dist = max-min

        #find maximum distance
        if dist > max_dist:
            max_dist = dist

        #set middle to 0,0,0
        np_pc[:,it] = np_pc[:,it] - dist/2 - min

        #covered cells
        cls = 29

        #find voxel edge size
        vox_sz = dist/(cls-1)

        #render pc to size 30x30x30 from middle
        np_pc[:,it] = np_pc[:,it]/vox_sz

    for it in range(0,3):
        np_pc[:,it] = np_pc[:,it] + (cls-1)/2

    #round to integer array
    np_pc = np.rint(np_pc).astype(np.uint32)

    #fill voxel arrays
    vox = np.zeros([30,30,30])
    for (pc_x, pc_y, pc_z) in np_pc:
        if random.randint(0,100) < 80:
            vox[pc_x, pc_y, pc_z] = 1

    if rot is not None:
        a = 1
        #TODO

    np_vox = np.zeros([1,1,32,32,32])
    np_vox[0, 0, 1:-1, 1:-1, 1:-1] = vox

    return np_vox

def voxel_scatter(np_vox):
    """

    this function

    Args:
        np_vox: nummpy ndarray of 5 dimensions with voxel volume at [~,~,x,y,z]

    Returns:
        numpy ndarray of num points by 3 that can be plotted by matplotlib scatter plot
    """
    #initialize empty array
    vox_scat = np.zeros([0,3], dtype= np.uint32)

    #itterate through x-dimensions
    for x in range(0,np_vox.shape[2]):
        #itterate through y-dimension
        for y in range(0,np_vox.shape[3]):
            #itterate through z-dimension
            for z in range(0,np_vox.shape[4]):
                #if voxel is dense add to scatter array
                if np_vox[0,0,x,y,z] == 1.0:
                    arr_tmp = np.zeros([1,3],dtype=np.uint32)
                    arr_tmp[0,:] = (x,y,z)
                    vox_scat = np.concatenate((vox_scat,arr_tmp))
    return vox_scat

class detector_voxnet:
    """

    use model_voxnet predict

    """
    def __init__(self, weights, nb_classes = 39):
        """

        Args:
            weights: keras weights file for voxnet, hdf5 type
            nb_classes: number of classes that the model was trained with

        Initializes the voxnet_model from model_keras with the given weights and classes, ready for detection

        """
        #initialize model_keras voxnet
        self.mdl = model_keras.model_vt(nb_classes=nb_classes, dataset_name="modelnet")

        #load weights into model_keras voxnet
        self.mdl.load_weights(weights)

    def predict(self, X_pred, is_pc = False):
        """

        Args:
            X_pred: Input Object nd.array of min 3 dimensions either voxnet size or other density cloud size
            is_pc: if not voxnet size (1x1x32x32x32) this has to be set to true

        Returns:
            returns the label name of the detected object, currently only works for objects from modelnet40 set
            return probability of which the detector puts in the detected object

        this will predict the probability for every given Object and pool the results and return the label
        and achieved probability

        """
        #if not voxnet size voxelize
        if is_pc == True:
            X_pred = voxilize(X_pred)

        #predict label and probability for all classes
        proba_all =  self.mdl.predict(X_pred)

        #retriev label for class with highest probability
        #indices 0 is equal to class 2
        label = str(np.argmax(proba_all) + 2)

        #turn label into string
        label = class_id_to_name_modelnet40[label]

        #retrieve value of probability
        proba = np.amax(proba_all)

        return label, proba