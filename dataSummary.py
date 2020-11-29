"""
## part summary for FreeCAD part library
date: 2019-10-07
"Mechanical Parts"  208MB, half is Fasteners
3 file types are given, stl, FCStd, step for each part/component
7000 data samples, imbalanced data

https://www.freecadweb.org/wiki/Fasteners_Workbench

text book: mechanial design parts

part identification, matching (hash)
==================
### usable
Fasteners
Cylinders
Pulleys

### not enough samples
 cable-chain-links
 coupling

### not single part, but assembly
 Eclosures
 Bearings
 Motor-CC
 Motedis
 Racor

 """

import os.path
import json
import numpy as np
import pandas as pd
from dataCollector import MechanicalOnly, dataset_filename, root_path

#dataset_filename = "inputData.json"
datafile = open(dataset_filename, "r")
dataList = json.loads(datafile.read())
data = pd.DataFrame(dataList)
print(data.head(10))
print(data.shape)
print(data.category.value_counts())

if MechanicalOnly:
    #get Fasteners only and subcategories
    fasteners = data[data.category == "Fasteners"]
    print(fasteners.shape)
    # it is not natural is deal with list as value
    has_subcat = fasteners.subcategories.map(len)>0
    #has_subcat.head()

    fasteners = fasteners[has_subcat]
    print(fasteners.shape)
    print(fasteners.head())
    fasteners["subtype"] = fasteners["subcategories"].map(lambda l: l[0])

    print(fasteners.shape)
    print(fasteners.subtype.value_counts())
    outputData = fasteners
else:
    outputData = data

dataFullPath = root_path + outputData["path"] + os.path.sep + outputData["filename"] + "\n"
with open("datapathfile.txt", "w") as f:
    for fp in dataFullPath:
        f.write(fp)

# list is unhashable
#print(data.subcategories.value_counts())
