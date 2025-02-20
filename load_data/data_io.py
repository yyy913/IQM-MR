import os
import numpy as np
from tqdm import tqdm

def save_data(data_dict, folder_name):
    root_dir = os.path.join(os.getcwd(), 'data')
    output_dir = os.path.join(root_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True) 
    for key, volume in tqdm(data_dict.items(), desc="Save volumes"):
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, volume)

def load_data_dict(folder_name):
    root_dir = os.path.join(os.getcwd(), 'data')
    output_dir = os.path.join(root_dir, folder_name)

    data_dict = {}

    for file_name in os.listdir(output_dir):
        if file_name.endswith('.npy'):
            key = os.path.splitext(file_name)[0]
            data_dict[key] = np.load(os.path.join(output_dir, file_name), allow_pickle=True)

    print(len(data_dict))
    return data_dict