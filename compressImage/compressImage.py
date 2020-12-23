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


def testCppyy():
    bs = b"123" #print ascii value ord()
    bs = array.array('B', range(16))
    cppyy.gbl.testBytes(bs, len(bs))  # test passed,  equally for  unsigned char *,  unsigned char []

    a = np.zeros((2,4), dtype=np.uint8)
    cppyy.gbl.testArray(bs, 8)

def compressImageBlock(im, block_size=8):
    #cppyy.gbl.std.vector[]   # how to get the result back
    #pb = cppyy.ll.cast["void *"](im.tobytes())  # does not work!

    assert(im.dtype == np.uint8)
    buf = im.tobytes()  # bytes type and content is fine
    #buf = copy.copy(buf)
    #print("bytes contents = ", buf)

    #pbuf = im.ctypes.data_as(ctypes.c_ubyte* len(buf))
    # TypeError: cast() argument 2 must be a pointer type, not c_ubyte_Array_4096
    #pbuf = ctypes.create_string_buffer(len(buf), buf)

    # ctypes.POINTER(ctypes.c_ubyte)  is for byte** output parameter,
    #pbuf = im.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte*len(buf))) #  error

    #pbuf = array.array('B', buf)  # correct
    pbuf = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)  # correctly
    #pbuf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * len(buf)))[0]  # correctly
    # from_buffer(buf)  TypeError: underlying buffer is not writable

    #print("type and size of pbuf ", type(pbuf), len(pbuf))
    #print("pbuf: ", pbuf[:16])

    h, w = im.shape[0]//block_size, im.shape[1]//block_size
    arr = cppyy.gbl.compressBlock(pbuf, im.shape) # .reshape((h*w,))
    #print(type(arr))
    cim = np.frombuffer(arr, dtype=np.uint64, count=h*w)  # count is not needed once reshape() is called
    cim = np.reshape(cim, (h, w))  #  cim = cim.reshape()
    #print(cim.shape)
    return cim


def compressImage(im, block_size=8, normalizing=False):
    """
    binary image should be multiple of 8, padding with zero or resize before call this
    compress a block_size X block_size block into a pixel of unsigned integer
    pure python implementation, slow, served as unit test for C version
    """
    height, width = im.shape
    if block_size==4:
        dtype = np.uint16
        middle = 2**16/2.0
    elif block_size==8:
        dtype = np.uint64
        middle = 2**63*1.0
    else:
        raise Exception("block size must be 4 or 8")

    if normalizing:
        result_dtype = np.float64
    else:
        result_dtype = dtype

    hBlock = height//block_size if height%block_size == 0 else height//block_size + 1
    wBlock = width//block_size if width%block_size == 0 else width//block_size + 1
    cim = np.zeros((hBlock, wBlock), dtype=result_dtype)

    lightThreshold = 30  # out of 255
    for y in range(hBlock):
        for x in range(wBlock):
            slice = im[y*block_size: (y+1)*block_size, x*block_size: (x+1)*block_size]
            bindata = np.zeros((block_size, block_size), dtype=np.uint8)
            bindata[slice>lightThreshold]= 1
            data = np.packbits(bindata)
            # todo: byte order !!!!!!!!!!!!!!!!!!!!!
            a = np.frombuffer(data.tobytes(), dtype=dtype)  # result is a 1D array
            if normalizing:
                cim[y, x] =  (a[0] - middle)/middle # normalized into [-1, 1)
            else:
                cim[y, x] =  a[0]
    return cim

"""
def compressImage(im):
    # np.packbits

    #N = IM_WIDTH * IM_HEIGHT / 8
    #im.reshape((N, 8))
    data = np.packbits(im)  # axis=None  (will flatten the ndarray first then), bitorder="big"
    # 2D binary image will be flatten into 1D array of unit8  by packbits

    #convert = lambda n : [int(i) for i in n.to_bytes(4, byteorder='big', signed=True)]   # in python3
    #np.frombuffer(b'\x01\x02', dtype=np.uint32)    #x.tobytes('C')
    return data
"""


def test():
    a = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,1], [0,0,0,0]])
    #print(a.shape)
    d = np.packbits(a)
    assert(d[0] == 128  and d[1] == 16)

    testCppyy()

    im=np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0],
                             [1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    im = im*128
    # todo: endianess is different
    cim1=compressImage(im.copy())
    cim2 = compressImageBlock(im)
    print(hex(cim1[0][0]), hex(cim2[0][0]))

if __name__ == "__main__":
    test()