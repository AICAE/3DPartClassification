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

if generatingThicknessViewImage:
    #image_suffices = ["_XY.csv", "_YZ.csv", "_ZX.csv"]  # thickness only
    image_suffices = ["_XY.png", "_YZ.png", "_ZX.png"]
else:
    image_suffices = ["_0.png", "_1.png", "_2.png"]


def process_image(image_stem, image_suffices, metadata):

    ims = collectImages(image_stem, image_suffices)  # it is quick enough

    # merge into a 2-channel image, must not do padding
    #dist_name = "_nearest"
    # dist_suffices = [dist_name + s for s in image_suffices]
    # if all([os.path.exists(image_stem+s) for s in dist_suffices]):
    #     assert not paddingImage
    #     dist_ims = collectImages(image_stem, dist_suffices)
    #     ims = np.dstack((ims, dist_ims))

    if isinstance(ims, (np.ndarray, np.generic)):
        imagelist.append(ims)
        dataset.append(metadata)  # image collection may fail

    # try:
    #     ims = collectImages(image_stem, image_suffices)  # it is quick enough
    #     if isinstance(ims, (np.ndarray, np.generic)):
    #         imagelist.append(ims)
    #         dataset.append(metadata)  # image collection may fail
    # except Exception as e:
    #     print(e)
    #     print("error in processing image", image_stem)

if dataset_name == "Thingi10K":
    CATEGORY_LABEL="Category"
    FILENAME_LABEL="File ID"

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
            process_image(image_stem, image_suffices, entry)
        else:
            print("Can not find all view image files for the input ", image_stem)


else:  # FreeCADLib,  or  ModelNet
    CATEGORY_LABEL="category"
    FILENAME_LABEL="filename"
    columns = ["filename", "category", "subcategories", "path"]

    def process_entry(entry):
        filename = entry["filename"]
        category = entry["category"]

        if dataset_name == "FreeCAD_lib":
            #del  metadata["center"]
            if splittingFastenerCategory and category == "Fasteners":
                entry["category"] = entry["subcategories"][0]
            # tmp hack
            #metafolder = "/mnt/windata/MyData/freecad_library_output" + os.path.sep + entry["path"]
            metafolder = output_root_path + os.path.sep + entry["path"]
        else:
            metafolder = output_root_path + os.path.sep + entry["path"]
        filefolder = output_root_path + os.path.sep + entry["path"]
        assert os.path.exists(filefolder)
        input_file_stem = filename[:filename.rfind('.')]                

        #images = glob.glob(filefolder + os.path.sep + input_file_stem +"*"+image_suffix)
        image_stem = filefolder + os.path.sep + input_file_stem
        all_image_found = all([os.path.exists(image_stem + s) for s in image_suffices])


        metadata_filename = glob.glob(metafolder + os.path.sep + input_file_stem + "*"+ metadata_suffix)
        if metadata_filename and all_image_found:
            f=open(metadata_filename[0], "r")
            metadata = json.loads(f.read())
            metadata["filename"] = filename

            metadata["category"] = entry["category"]
            if "subcategories" in entry:
                metadata["subcategories"] = entry["subcategories"]
            else:
                metadata["subcategories"] = []

            process_image(image_stem, image_suffices, metadata)
        else:
            # about 10 can not find meta data files,  image dump may have error
            print("Can not find metadata or all view image files ", input_file_stem)


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
    print("processed_metadata_filepath = ", processed_metadata_filepath)
    print("save processed image file as ", processed_imagedata_filename)
    print("single processed image shape ", imagelist[0].shape)

