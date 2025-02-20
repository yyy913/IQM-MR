import os
import numpy as np
from tqdm import tqdm
from load_data.load_from_dropbox import dropbox_connect, load_npy_from_dropbox
from load_data.data_io import save_data

def process_group(file_keys, base_path='/simulation_data/'):
    dbx = dropbox_connect()
    for key in file_keys:
        file_path = f"{base_path}{key}"
        data = load_npy_from_dropbox(dbx, file_path)
        print(f"Loaded {len(data)} npy files for {key}.")
        save_data(data, key)

if __name__ == "__main__":
    groups = {
        "t1post": ["t1post_clear", "t1post_g1", "t1post_g2", "t1post_g3", "t1post_g4", "t1post_g5"],
        "t1":     ["t1_clear", "t1_g1", "t1_g2", "t1_g3", "t1_g4", "t1_g5"],
        "t2":     ["t2_clear", "t2_g1", "t2_g2", "t2_g3", "t2_g4", "t2_g5"],
        "flair":  ["flair_clear", "flair_g1", "flair_g2", "flair_g3", "flair_g4", "flair_g5"]
    }

    for group_name, file_keys in groups.items():
        process_group(file_keys)