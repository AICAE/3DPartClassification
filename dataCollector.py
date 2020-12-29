"""
For FreeCAD library parts dataset and Thingi10K dataset
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

dataset = []
imagelist = []

def process_image(image_stem, image_suffices):
    try:
        ims = collectImages(image_stem, image_suffices)  # it is quick enough
        if isinstance(ims, (np.ndarray, np.generic)):
            imagelist.append(imagelist, ims)
            dataset.append(metadata)  # image collection may fail
    except Exception as e:
        print(e)
        print("error in processing image", image_stem)

if datasetName == "Thingi10K":
    CATEGORY_LABEL="Category"
    FILENAME_LABEL="File ID"
    image_suffices = ["_XY.csv", "_YZ.csv", "_ZX.csv"]

    def process_entry(entry):
        # metadata has been collected and merged, still copy to dataset list to write as pandas.DataFrame
        fileid = entry[FILENAME_LABEL].split(" ")[0]  # "File ID": "100034 (Download)"
        entry["fileid"] = fileid

        filefolder = output_root_path
        assert os.path.exists(filefolder)
        input_file_stem = fileid           

        image_stem = filefolder + os.path.sep + input_file_stem
        # check: return NoneType /mnt/windata/MyData/Thingi10K_dataset_output/196196
        if all([os.path.exists(image_stem + s) for s in image_suffices]):
            process_image(image_stem, image_suffices)
        else:
            #print("Can not find image files for ", image_stem)
            pass

else:  # FreeCADLib
    CATEGORY_LABEL="category"
    FILENAME_LABEL="filename"
    columns = ["filename", "category", "subcategories", "path"]
    image_suffices = ["_0.png", "_1.png", "_2.png"]

    def procoss_entry(entry):
        filename = entry["filename"]
        catetory = entry["category"]
        #"subcategories"
        filefolder = output_root_path + os.path.sep + entry["path"]
        assert os.path.exists(filefolder)
        input_file_stem = filename[:filename.rfind('.')]                

        images = glob.glob(filefolder + os.path.sep + input_file_stem +"*"+image_suffix)
        image_stem = filefolder + os.path.sep + input_file_stem

        metadata_filename = glob.glob(filefolder + os.path.sep + input_file_stem + "*"+ metadata_suffix)
        if metadata_filename and images:
            f=open(metadata_filename[0], "r")
            metadata = json.loads(f.read())
            metadata["filename"] = filename
            del  metadata["center"]
            metadata["category"] = entry["category"]
            metadata["subcategories"] = entry["subcategories"]

            process_image(image_stem, image_suffices)
        else:
            # about 10 can not find meta data files,  image dump may have error
            print("Can not find metadata or image files ", input_file_stem)


def process():
    f=open(dataset_metadata_filepath, "r")
    REGISTRY = json.loads(f.read())  # could be dict or list
    if isinstance(REGISTRY, dict):
        REGISTRY = REGISTRY.values()

    part_count = len(REGISTRY) # list type
    print("total part registered: ", part_count)

    count = 0
    step_count =100
    step_i = 0
    pbar = tqdm(total=step_count, file=sys.stdout)

    for entry in REGISTRY:
        process_entry(entry)
        step_i +=1
        if step_i== int(part_count/step_count):
            pbar.update(1)
            step_i=0
        count +=1

        #if count > 10:        break
    print("registered {} items", len(imagelist))

if __name__ == "__main__":

    process()
    if (dataset):
        dataframe = pd.DataFrame(dataset)
        print(dataframe[:5])  # debug print
        dataframe.to_json(processed_metadata_filepath)
    #save images to binary format, for quicker loading
    np.save(processed_imagedata_filename, np.stack(imagelist))

