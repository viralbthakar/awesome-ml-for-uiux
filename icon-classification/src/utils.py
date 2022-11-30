import os
import glob
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_dir(root, new_dir=''):
    os.makedirs(os.path.join(root, new_dir), exist_ok=True)
    return os.path.join(root, new_dir)


def load_npy_file(filepath, summary=True):
    data = np.load(filepath)
    unique, counts = np.unique(data, return_counts=True)
    if summary:
        print("# - "*5, f"{os.path.basename(filepath)}", " - #"*5)
        print(f"Shape of numpy array: {data.shape}")
        print(f"Min value: {np.amin(data)}")
        print(f"Max value: {np.amax(data)}")
        print(f"Found {len(unique)} unique values.")
    return data, unique, counts


def load_npy_files(filepaths, summary=True, freq_count=True):
    data = {
        "data": [],
        "unique": [],
        "counts": []
    }
    if isinstance(filepaths, list):
        for filepath in filepaths:
            arr, unique, counts = load_npy_file(filepath, summary=summary)
            data["data"].append(arr)
            data["unique"].append(unique)
            data["counts"].append(counts)
    else:
        arr, unique, counts = load_npy_file(filepath, summary=summary)
        data["data"].append(arr)
        data["unique"].append(unique)
        data["counts"].append(counts)
    return data


def extract_data_from_json(filepath, key):
    with open(filepath) as user_file:
        parsed_json = json.load(user_file)
    return parsed_json[key]


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


def plot_distribution(x, y, title="Distribution Plot", figsize=(24, 24), dpi=300,
                      save_flag=False, file_path=None, orient='h',
                      palette="one-color", xticks_ct=10):
    if palette == "one-color":
        palette = sns.color_palette(n_colors=1)
    else:
        palette = sns.color_palette(palette)

    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    pt = sns.barplot(x=x, y=y, palette=palette, orient=orient)

    for ind, label in enumerate(pt.get_xticklabels()):
        if isinstance(ind, int) or isinstance(ind, float):
            if int(ind % xticks_ct) == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        else:
            label.set_visible(True)

    plt.title(title)
    if save_flag:
        fig.savefig(file_path, dpi=dpi, facecolor='white')
        plt.close


def create_validation_split(data_generator, validation_split=0.2, shuffle=True, seed=1):
    dataset_size = sum(1 for _ in data_generator)
    if shuffle:
        data_generator = data_generator.shuffle(dataset_size, seed)

    train_dataset_size = int(dataset_size * (1 - validation_split))

    train_dg = data_generator.take(train_dataset_size)
    val_dg = data_generator.skip(train_dataset_size)
    return train_dg, val_dg
