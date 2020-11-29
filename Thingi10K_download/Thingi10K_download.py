
"""
Auto-generated download script for Thingi10K dataset.
Assuming the following python packages and external commands are available.

* argparse: for parse command line args.
* requests: for http communication.
* wget: for downloading files.

Usage:

    python Thingi10K_download.py

or

    python Thingi10K_download.py -o output_dir

"""

import argparse
import os.path
import sys
from subprocess import check_call
from file_ids_list import file_ids
#file_ids = ["34785"]   # testing purpose

from Thingi10K_download_metadata import download_metadata

output_dir = "/mnt/windata/MyData/Thingi10K_dataset/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


try:
    import requests
except ImportError:
    error_msg = "Missing module: requests.  To fix it, please run '$ pip install requests'";
    sys.exit(error_msg);

try:
    check_call("which wget".split());
except:
    error_msg = "Missing command: wget.  To fix it, please run '$ port install wget'";
    sys.exit(error_msg);


def has_downloaded(file_id, output_dir):
    output_file = os.path.join(output_dir, "{}.stl".format(file_id));
    return os.path.exists(output_file )


def download_file(file_id, output_dir):
    if not os.path.isdir(output_dir):
        raise IOError("Directory {} does not exists".format(output_dir));
    url = "https://www.thingiverse.com/download:{}".format(file_id);
    r = requests.head(url);
    link = r.headers.get("Location", None);
    if link is None:
        print("File {} is no longer available on Thingiverse.".format(file_id));
        return;

    __, ext = os.path.splitext(link);
    output_file = "{}{}".format(file_id, ext.lower());
    output_file = os.path.join(output_dir, output_file);
    print("Downloading {}".format(output_file));
    command = "wget -q -O {} --tries=10 {}".format(output_file, link);
    check_call(command.split());


def main():
    error_list = []
    for file_id in file_ids:
        if not has_downloaded(file_id, output_dir):
            try:
                download_metadata(file_id, output_dir)
                download_file(file_id, output_dir)
            except:
                error_list.append(file_id)
    if error_list:
        json.dump(error_list, open("error_fileid_list.json", "w"))

if __name__ == "__main__":
    main();
