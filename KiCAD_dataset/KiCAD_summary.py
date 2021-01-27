import os.path
import os
import glob

groups = {}
data_folder = "/mnt/windata/DataDir/kicad-packages3D"
threshold = 40

# merge package
# plot images in grid?

subFolders = [o for o in os.listdir(data_folder) if os.path.isdir(data_folder + os.path.sep + o)]
for d in sorted(subFolders):  # recursive subfolder processing
    input_dir = data_folder + os.path.sep + d
    if d.endswith(".3dshapes"):
        gname = d.split(".")[0]
        step_files = glob.glob(input_dir + os.path.sep + "*.step")
        if len(step_files) > threshold:       
            groups[gname] = len(step_files)

print(groups)
