from __future__ import print_function, division
import subprocess
import os
import os.path
import sys
import time
import random

import numpy as np
from matplotlib import pyplot as plt
import cv2

from compressImage import compressImageBlock as compressImageCpp
from compressImage import compressImage

from fclib_parameters import *
#IM_WIDTH, IM_HEIGHT = 128, 128
#block_size = 8

if block_size==4:
    block_dtype = np.uint16
    middle = 2**16/2.0
elif block_size==8:
    block_dtype = np.uint64
    middle = 2**63*1.0
else:
    raise Exception("block size must be 4 or 8")

def collectImages(filename_stem):
    filenames = [filename_stem + s for s in ["_0.png", "_1.png", "_2.png"]]
    images = []

    for f in filenames:
        img = cv2.imread(f, 0)     # Load an color image in grayscale
        #img = rgb2gray(img)
        img = cv2.resize(img,(im_width, im_height))
        #print("after resize: ", img.shape)
        if binarizingImage and not compressingImage:
            th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,11,2)
        if paddingImage:
            # 3 view images should have the same padding offset?
            #print("before padding: ", img.shape)
            img = padImage(img, (IM_WIDTH, IM_HEIGHT))

        if compressingImage:
            img = compressImageCpp(img, block_size=block_size)
        images.append(img)

    filename = os.path.basename(filename_stem)
    if concatingImage:
        return concatImages(images, filename)
    else:
        return images

def concatImages(images, filename):

    # full filename  = root + filename suffix `_X.png`
    #concat images
    im = cv2.hconcat(images)
    # debug show image
    #showImage(im, title=filename)
    return im

def needPadding(im, block_size=8):
    height, width = im.shape
    if height%block_size == 0 and width%block_size == 0:
        return False
    else:
        return True

def padImage(im, result_shape, padding_value=0):
    # also apply random shift
    #cv2.copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );

    height, width = im.shape
    yPad = result_shape[0] - height
    xPad = result_shape[1] - width
    assert yPad >=0 and xPad >= 0
    #randint, to get start
    yStart = random.randint(0, yPad)
    xStart = random.randint(0, xPad)

    rim = np.full(result_shape, padding_value, dtype=im.dtype)
    rim[yStart: yStart+height, xStart: xStart+width] = im
    #numpy put works only for flatten array
    return rim

def rgb2gray(rgb):
    # input, 3 channels, output must be single chanel uint8,
    #print(rgb.dtype, rgb[0,0])
    im = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    #print(im.dtype, im[0,0])
    img = np.zeros(im.shape, dtype=np.uint8)
    img[:,:] = im*256
    #print(img.dtype, img[0,0])
    return img

def binarizeImage(im):
    # even bool_ does not mean use only 1bit storage memory
    bim = np.zeros_like(im, dtype = np.bool_)
    # specify a threshold 0-255
    threshold = 150
    bim[ im > threshold ] = 1
    return bim

def normalizeImage(im, block_size=8):
    result_dtype = np.float64
    img = np.zeros(im.shape, dtype=result_dtype)
    img = (im - middle)/middle # normalized into [-1.0, 1)
    return img

def showImage(im, title):
    fig = plt.figure()
    plt.imshow(im)
    plt.title(title)


def run_command(comandlist):
    has_error = False
    try:
        print(comandlist)
        p = subprocess.Popen(comandlist, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p.communicate()
        print(output)  # stdout is still cut at some point but the warnings are in stderr and thus printed :-)
        print(error)  # stdout is still cut at some point but the warnings are in stderr and thus printed :-)
    except:
        print('Error executing: {}\n'.format(comandlist))
        has_error = True
    return has_error

def test():
    imfile = "testdata/testimage.png"
    im = plt.imread(imfile)  

    im = rgb2gray(im) # 8bit grayscale
    print("rgb2gray(im): ", im.shape, im.dtype)
    #showImage(im, title="grayscale")  # still show as color

    paddedShape = [64, 64] # should be about 110% of the input
    im = padImage(im, paddedShape)
    #showImage(im, title="padd")
    print("first block", im[:block_size , :block_size])  # first block
    #im = binarizeImage(im)  # binarization will be done in image compression

    cim = compressImageCpp(im)
    #print(cim)  # should be lots of zeros, !!!
    img = normalizeImage(cim)  # fine, normalized into order to show image
    #print(img)
    showImage(img,  title = "compressed image in C")

    cim2 = compressImage(im)  # numpy
    img2 = normalizeImage(cim2)
    showImage(img2,  title = "compressed image in Python")

    plt.show()

if __name__ == "__main__":
    test()