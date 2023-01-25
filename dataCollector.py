"""
after DTV images has been generated, collect into
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

from global_config import *
from input_parameters import dataset_name, output_root_path, dataset_metadata_filepath, metadata_suffix, \
      generatingThicknessViewImage, processed_imagedata_filepath, processed_metadata_filepath
from imagePreprocess import collectImages

# here there is a bug output_root_path has not been updated, due to import sequence, or multiple import
if dataset_name == "FreeCAD_lib":
    from fclib_parameters import *
    output_root_path = "/mnt/DataDir/freecad_library_output_thickness/"

if dataset_name == "KiCAD_lib":
    from kicad_parameters import *
    output_root_path = "/media/DataDir/kicad-packages3D_output_thickness/"

dataset = []
imagelist = []

if generatingThicknessViewImage:
    #image_suffices = ["_XY.csv", "_YZ.csv", "_ZX.csv"]  # thickness only
    image_suffices = ["_XY.png", "_YZ.png", "_ZX.png"]
    if usingRotatingZAxisView:
        imagelist += ["_RZ_XY.png", "_RZ_YZ.png", "_RZ_ZX.png"]
    if usingTriView:
        imagelist += ["_TRI_XY.png", "_TRI_YZ.png", "_TRI_ZX.png"]
else:
    image_suffices = ["_0.png", "_1.png", "_2.png"]


def process_image(image_stem, image_suffices, metadata):
    if "bbox" in metadata:
        bbox = metadata["bbox"]
        if any([ bbox[i+3] - bbox[i] <= 0 for i in range(3)]):
            print("WARNING: Boundbox zero for ", image_stem)
            return  # skip for this kind of error
    if (metadata["area"] <= 0 or metadata["volume"] <= 0):
        print("WARNING: zero area or volume for ", image_stem)
        return

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


else:  # FreeCADLib, KiCAD or  ModelNet, ShapeNet
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
            if splittingProfileCategory and category == "Profiles EN":
                entry["category"] = entry["subcategories"][0]

            if category == "Profiles EN": # nasty patch, after reorganized processed image folder
                for p in ['DIN1025-2 HE-B-Profiles',  'DIN1025-4 HE-M-Profiles', 'DIN1025-3 HE-A-Profiles', 'DIN1025-5 IPE-Profiles']:
                    if p in entry["path"]:
                        entry["path"] = entry["path"].replace(p, "DIN1025-Profiles" + os.path.sep + p)
                        entry["category"] = "DIN1025-Profiles"
                for p in ['EN10056 Equal Angle Bars',  'EN10056 Unequal Angle Bars']:
                    if p in entry["path"]:
                        entry["path"] = entry["path"].replace(p, "EN10056 Angle Bars" + os.path.sep + p)
                        entry["category"] = "EN10056 Angle Bars"

        if dataset_name == "KiCAD_lib":
            #category = entry["category"].split('.')[0]
            # specific dataset, filitering and adapting
            
            #if merging_SMD_THT and category.startswith('Inductor'):
            #    entry["category"] = 'Inductor'
            if category.startswith('Inductor_SMD'):
                return
            if merging_SMD_THT and category.startswith('Capacitor'):
                entry["category"] = 'Capacitor'
            if merging_SMD_THT and category.startswith('Button_Switch'):
                entry["category"] = 'Button_Switch'

        if dataset_name == "ShapeNetCore":
            # will category equal to subfolder name?
            entry["category"] = 'Capacitor'

        # from relative path to abspath
        filefolder = output_root_path + os.path.sep + entry["path"]
        metafolder = output_root_path + os.path.sep + entry["path"]
        #print("filefolder = ", filefolder + os.path.sep + filename)
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
            print("Can not find metadata or all view image files ", filefolder + os.path.sep + input_file_stem)

def filter(dataframe, imagelist, class_col_name = "category", minimum_sample_count = minimum_sample_count):
    # remove group/class whose sample count is less than a threshold
    df = pd.DataFrame(dataset)  # natural index?
    # add Index to dataframe, then group, get the unique classes
    df = df.reset_index()
    df['orig_index'] = df.index   # images has the same order as df
    df.groupby([class_col_name])
    #group_values = df[class_col_name].unique()  # return value list?
    table = df.groupby([class_col_name]).size().reset_index(name='counts')
    table = table[table['counts'] >= minimum_sample_count]
    classes = table[class_col_name]
    df_filtered = df[df[class_col_name].isin(classes)]

    ## update image list squence to match sorted table df
    imglist = []
    # use `values`` instead of `to_numpy()` AttributeError: 'Series' object has no attribute 'to_numpy'
    for i in df_filtered['orig_index'].values:
        imglist.append(imagelist[i])
    return df_filtered, imglist

def summary(df, class_col_name = "category"):
    table = df.groupby([class_col_name]).size().reset_index(name='counts')
    print(table)
    print("dataset sample summary", dataset_name)
    try:
        print(table.to_markdown()) # new API since pandas 1.0
    except:
        pass

def save(dataset, imagelist):
    dataframe, imagelist = filter(pd.DataFrame(dataset), imagelist)
    summary(dataframe)
    print(dataframe[:5])  # debug print
    dataframe.to_json(processed_metadata_filepath)
    #save images to binary format, for quicker loading
    np.save(processed_imagedata_filepath, np.stack(imagelist))
    print("processed_metadata_filepath = ", processed_metadata_filepath)
    print("save processed image file as ", processed_imagedata_filepath)
    print("single processed image shape ", imagelist[0].shape)

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

    print("registered {} items", len(imagelist))

if __name__ == "__main__":

    process()
    if (dataset):
        save(dataset, imagelist)

