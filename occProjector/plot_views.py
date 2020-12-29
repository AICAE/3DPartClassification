
import numpy as np 
from matplotlib import pyplot as plt
import sys
import glob

default_input_stem = "data/part.stl"
default_input_stem = "data/part.brep"
#default_input_stem = "data/part.brep_BOP"
suffixes = ["_XY.csv", "_YZ.csv", "_ZX.csv"]

#default_input_stem = "/mnt/windata/MyData/OneDrive/gitrepo/PartRecognition/testmesh_output/chair/train/chair_0893.stl"

#if bop:
#   suffixes = ["_BOP" + s for s in suffixes]


def plot_projection_views(outputfile_stem):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Vertically stacked subplots')

    #input_files = glob.glob(outputfile_stem + "*.csv")
    input_files = [outputfile_stem + s for s in suffixes]

    for i, f in enumerate(input_files):
        im = np.loadtxt(f, delimiter=',')
        #normalize,  because min is zero, to
        immax = np.max(im) # some point, value to too big, NAN, that scaling to all zeros
        print(i, immax, f)
        im = im/immax
        axs[i].imshow(im)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        outputfile_stem = default_input_stem
    else:
        outputfile_stem = sys.argv[1]

    plot_projection_views(outputfile_stem)
