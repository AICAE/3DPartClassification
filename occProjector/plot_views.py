
import numpy as np 
from matplotlib import pyplot as plt
import sys
import glob

default_input_file = "data/part.stl"
default_input_file = "/mnt/windata/MyData/OneDrive/gitrepo/PartRecognition/testmesh_output/chair/train/chair_0893.stl"
# bop = False
# suffixes = ["_XY.csv", "_YZ.csv", "_ZX.csv"]
# if bop:
#     suffixes = ["_BOP" + s for s in suffixes]


def plot_projection_views(outputfile_stem):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Vertically stacked subplots')

    input_files = glob.glob(outputfile_stem + "*.csv")
    # input_files = [outputfile_stem + s for s in suffixes]

    for i, f in enumerate(input_files):
        im = np.loadtxt(f, delimiter=',')
        #normalize,  because min is zero, to
        immax = np.max(im) # some point, value to too big, NAN, that scaling to all zeros
        print(i, immax)
        im = im/immax
        axs[i].imshow(im)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        inputfile = default_input_file
        outputfile_stem = inputfile
    else:
        outputfile_stem = sys.argv[1]
    plot_projection_views(outputfile_stem)
