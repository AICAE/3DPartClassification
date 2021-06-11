import os.path
import os

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from input_parameters import dataset_name, saved_model_filepath
result_dir = 'results'

# class-wise accuracy table, -> confusion matrix

#saved_model_filepath = r"D:\DataDir\ModelNet10_output_thickness\6views_DT_cubebox_mixedinput_ModelNet10_feb12.h5"

saved_model_name = os.path.basename(saved_model_filepath)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
dot_img_file = f'{result_dir}/{saved_model_name}_model.png'
model = tf.keras.models.load_model(saved_model_filepath)
# failed on windows, becaused dot.exe is not installed
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model.summary()

# save to json:
hist_json_files = [saved_model_filepath + '.json']
frames = []
for i, jf in enumerate(hist_json_files):
    frames.append(pd.read_json(jf))

history = pd.concat(frames)
loss_figure_file = f'{result_dir}/{saved_model_name}_loss.png'

# using history can plot val_accurary (validation accurary)
plt.plot(history['accuracy'], "-b", label='accuracy')
plt.plot(history['val_accuracy'], ".k", label = 'val_accuracy')
plt.plot(history['loss'], "r--", label='loss')
plt.plot(history['val_loss'], "g.-", label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0, 1])
plt.title(saved_model_name)
plt.legend(loc='lower right')

plt.show()
plt.savefig(loss_figure_file)


#tf.math.confusion_matrix(    labels, predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,     name=None )