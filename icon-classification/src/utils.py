import os
import glob
import json


def list_files_from_dir(path, extension='*.*', recursive=False, sort=True):
    if recursive:
        dir_path = os.path.join(path, "**", extension)
    else:
        dir_path = os.path.join(path, extension)
    files = glob.glob(dir_path)

    if sort:
        files.sort()
    return files


def extract_value_for_key(data_dict, key):
    if key in data_dict.keys():
        return data_dict[key]
    else:
        return None
