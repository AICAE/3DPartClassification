
import numpy as np 
from matplotlib import pyplot as plt
import sys
import glob
import os.path
from mpl_toolkits.mplot3d import Axes3D


#default_input_stem = "data/part.stl"
default_input_stem = "data/part.brep"
if not os.path.isabs(default_input_stem) and os.path.exists("occProjector"):
    # may run with cwd = repo root folder
    default_input_stem = "occProjector/" + default_input_stem

#default_input_stem = "/mnt/windata/MyData/Thingi10K_dataset_output/36086"
#default_input_stem = "/mnt/windata/MyData/OneDrive/gitrepo/PartRecognition/testmesh_output/chair/train/chair_0893.stl"

bop = False
usingPNG = True
using3D = False  # very slow
usingTriView = True
vmin, vmax = 0.0, 1.0
view_names = ["_XY", "_YZ", "_ZX"]

if usingPNG:
    suffix = ".png"
    nrows = len(view_names)
    ncols = 3
    channel_names = ["depth", "thickness", "back_depth"]
else:
    suffix = ".csv"
    nrows = 1
    ncols = 3

if bop:
  view_names = ["_BOP" + s for s in view_names]
if usingTriView:
  view_names = ["_TRI" + s for s in view_names]
suffixes = [v + suffix for v in view_names]


def read_csv(f):
    im = np.loadtxt(f, delimiter=',')
    #normalize,  because min is zero, to
    immax = np.max(im) # some point, value to too big, NAN, that scaling to all zeros
    #print(immax, f)
    im = im/immax
    return im

def plot_projection_views(outputfile_stem):
    if using3D:
        fig = plt.figure(figsize=(24, 32))
    else:
        fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle('Vertically stacked subplots')

    #input_files = glob.glob(outputfile_stem + "*.csv")
    input_files = [outputfile_stem + s for s in suffixes]

    for i, f in enumerate(input_files):
        if f.endswith(".csv"):
            im = read_csv(f)
            axs[i].imshow(im)
        else:
            im = plt.imread(f)
            nx, ny = im.shape[0], im.shape[1]
            if len(im.shape) > 2:
                nchannels = im.shape[2]
            else:
                nchannels = 1
            for ch in range(nchannels):
                if using3D:
                    ax = fig.add_subplot(nrows, ncols, i*ncols+ch+1, projection='3d')
                    X = np.arange(vmin, vmax, (vmax - vmin)/nx)
                    Y = np.arange(vmin, vmax, (vmax - vmin)/ny)
                    X, Y = np.meshgrid(X, Y)
                    Z = im[:, :, ch]
                    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                       linewidth=0, antialiased=False)
                    ax.set_zlim(vmin-0.05, vmax+0.05)  # hide the thickness of zero, is that possible? histgram/bar?
                else:
                    ax = axs[i, ch]
                    ax.imshow(im[:, :, ch])
                    # todo: set colormap vmin and vmax to to same
                ax.set_axis_off()
                ax.set_title(view_names[i]+ " " + channel_names[ch])
                #ax.legend()

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        outputfile_stem = default_input_stem
    else:
        outputfile_stem = sys.argv[1]

    plot_projection_views(outputfile_stem)
