"""
For FreeCAD library parts dataset only
"""

from __future__ import print_function, division
import os
import os.path
import sys
import time
import json
import glob

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2g}'.format
from tqdm import tqdm

from input_parameters import *
from imagePreprocess import collectImages

def process():
    f=open(dataset_filename, "r")
    REGISTRY = json.loads(f.read())

    part_count = len(REGISTRY) # list type
    print("total part registered: ", part_count)

    columns = ["filename", "category", "subcategories", "path"]

    dataset = []

    imagelist = []

    image_suffices = ["_0.png", "_1.png", "_2.png"]
    count = 0
    step_count =100
    step_i = 0
    pbar =  tqdm(total=step_count, file=sys.stdout)

    for entry in REGISTRY:
        filename = entry["filename"]
        catetory = entry["category"]
        #"subcategories"
        filefolder = output_root_path + os.path.sep + entry["path"]
        assert os.path.exists(filefolder)
        file_stem = filename[:filename.rfind('.')]                

        images = glob.glob(filefolder + os.path.sep + file_stem +"*"+image_suffix)
        image_stem = filefolder + os.path.sep + file_stem

        metadata_filename = glob.glob(filefolder + os.path.sep + file_stem + "*"+ metadata_suffix)
        if metadata_filename and images:
            f=open(metadata_filename[0], "r")
            metadata = json.loads(f.read())
            metadata["filename"] = filename
            del  metadata["center"]
            metadata["category"] = entry["category"]
            metadata["subcategories"] = entry["subcategories"]

            imagelist.append(collectImages(image_stem, image_suffices))  # it is quick enough
            try:
                dataset.append(metadata)  # image collection may fail
            except Exception as e:
                print(e)
                print("error in processing image", image_stem)
        else:
            # about 10 can not find meta data files,  image dump may have error
            print("Can not find metadata for ", filename)

        step_i +=1
        if step_i== int(part_count/step_count):
            pbar.update(1)
            step_i=0
        count +=1

        #if count > 10:        break
    print("registered {} items", len(dataset))
    dataframe = pd.DataFrame(dataset)
    return dataframe, imagelist


if __name__ == "__main__":

    dataframe, imagelist = process()

    print(dataframe[:5])

    #save images to binary format, for quicker loading
    #s = pd.Series(["a", "b", "c", "a"], dtype="category")
    np.save(processed_imagedata_filename, np.array(imagelist))
    dataframe.to_json(processed_metadata_filename)
    #imagelist processed_imagedata