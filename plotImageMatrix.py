# this may not work
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from input_parameters import dataset_name, channel_count, thickness_channel, \
    processed_metadata_filepath, processed_imagedata_filepath, saved_model_filepath
from stratify import index_split

df = pd.read_json(processed_metadata_filepath)
images = np.load(processed_imagedata_filepath)  # pickle array of object type: allow_pickle=True
class_col_name = "category"
# add Index to dataframe, then group, get the unique classes
df = df.reset_index()
df['orig_index'] = df.index   # images has the same order as df
df.groupby([class_col_name])
cat_names = df[class_col_name].unique()  # return value list? 
cat_count = len(cat_names)

rows = 4 # cat_count
columns = 4
fig = plt.figure(figsize=(columns*2+1, rows*2+1))
axes = []

print("input image shape: ", images[0].shape)
i = 0
for r in range(rows):
    val = cat_names[r]
    group_data = df[df[class_col_name] == val]
    group_img_index = group_data['orig_index'].to_numpy() 
    group_size = len(group_data)
    for c in range(columns):
        img_index = group_img_index[c] # todo
        img = images[img_index][0, :, :, thickness_channel]
        ax = fig.add_subplot(rows, columns, i+1)
        axes.append(ax)
        if c == 0:
            ax.set_title(val)  # set title
        ax.set_axis_off()
        plt.imshow(img)
        # no grid
        i += 1


plt.show()  # finally, render the plot