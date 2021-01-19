from __future__ import print_function, division
import subprocess
import os
import os.path
import sys
import time
import random

import numpy as np
from matplotlib import pyplot as plt

import ctypes
import array
import copy
import cppyy
import cppyy.ll

cwd = os.path.dirname(os.path.abspath(__file__))
good = cppyy.cppdef(open(cwd + os.path.sep + "compressBlock.cpp").read())
if not good:
    print("cppyy failed to compile the cpp code")


def checkVoxelShape(block_size):
    """
    """
    if block_size==2:
        dtype = np.uint8
        middle = 2**7*1.0
        ctype = "uint8_t"
    elif block_size==3:
        dtype = np.uint32
        middle = 2**26*1.0
        ctype = "uint32_t"
    elif block_size==4:
        dtype = np.uint64
        middle = 2**63*1.0
        ctype = "uint64_t"
    else:
        raise Exception("block size must be 2, 3,4 for voxel compression")
    return dtype, middle, ctype


def compressVoxelBlock(im, block_size=2):
    """ """
    assert(im.dtype == np.uint8)
    dtype, middle, ctype = checkVoxelShape(block_size)

    buf = im.tobytes()  # bytes type and content is fine
    pbuf = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)  # correctly

    d, h, w = im.shape[0]//block_size, im.shape[1]//block_size, im.shape[2]//block_size
    arr = cppyy.gbl.compressVoxel[ctype](pbuf, im.shape)
    cim = np.frombuffer(arr, dtype=dtype, count=d*h*w)  # count is not needed once reshape() is called
    result = np.reshape(cim, (d, h, w))  #  cim = cim.reshape()
    #print(cim.shape)
    return result

def plot_voxel(voxels):
    # and plot everything
    from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='r', edgecolor='k')

    plt.show()

def test_compressVoxel():

    im=np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    vim = np.reshape(im, (4, 4, 4))
    #plot_voxel(vim)  # fine

    cim2 = compressVoxelBlock(vim, block_size=2)

    cim4 = compressVoxelBlock(vim, block_size=4)


if __name__ == "__main__":
    test_compressVoxel()