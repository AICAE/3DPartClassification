import os.path
import json
import numpy as np
import pandas as pd

from input_parameters import input_root_path, processed_metadata_filepath

def summarize(df, class_col_name = "category"):
    table = df.groupby([class_col_name]).size().reset_index(name='counts')
    print(table)
    print(table.to_markdown())
    
    #for label in ["area", "volume"]:
    print("== Table of max for groups ==")
    table = df.groupby([class_col_name]).max() #.reset_index(name='group_max')
    print(table)
    print(table.to_markdown())

    table = df.groupby([class_col_name]).mean() # .reset_index(name='group_mean')
    print("== Table of Mean for groups ==")
    print(table)

df = pd.read_json(processed_metadata_filepath)
summarize(df[["category", "volume", "area"]])