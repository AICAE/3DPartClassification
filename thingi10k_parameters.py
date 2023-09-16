
# this dataset has very bad training accuracy

from global_config import *

isMeshFile = True    # choose between  part and mesh input format
hasPerfileMetadata  = True

##############################
if testing:
    input_root_path = "./testdata/testThingi10K_data"
    output_root_path = input_root_path + "_output"
    dataset_metadata_filename =  "testThingi10K_dataset.json"
else:
    # only available with the developer, yet uploaded to repo
    input_root_path = DATA_DIR + "Thingi10K_dataset"
    output_root_path = DATA_DIR + "Thingi10K_dataset_output"
    dataset_metadata_filename = "Thingi10K_dataset.json"

def collect_metadata():
    # only works if all input files are in the same folder
    metadata = {}
    if hasPerfileMetadata:
        # no needed for input format conversion
        all_input_files = glob.glob(input_root_path  + os.path.sep  + "*." + metadata_suffix)
        #print(all_input_files)
    for input_file in all_input_files:
        #print("process metadata for input file", input_file)
        f = input_file.split(os.path.sep)[-1]
        fileid = f[:f.rfind(".")]
        with open(input_file, "r") as inf:
            m = json.load(inf)
        json_file_path = output_root_path + os.path.sep + str(fileid) + "_metadata.json"
        if(os.path.exists(json_file_path)):
            pm = json.load(open(json_file_path, "r"))
            for k, v in pm.items():
                m[k] = v
            metadata[fileid] = m
        else:
            print("Warning: there is no generated metdata file", json_file_path)
    return metadata

    # collect from  output folder