import dropbox
import os
from io import BytesIO
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import h5py

APP_KEY = 'wez2mo5zuap2e3d'
REFRESH_TOKEN = 'SU8aoNQ5Q_EAAAAAAAAAAafx4W1j_Dg0duCClZAivpPW-D7aFQJXWyII3dgtgwRh'

def dropbox_connect():
    dbx = dropbox.Dropbox(oauth2_refresh_token=REFRESH_TOKEN, app_key=APP_KEY)
    return dbx

def download_npy_file(dbx, dropbox_path, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Downloading {dropbox_path}, attempt {attempt + 1}")
            _, response = dbx.files_download(dropbox_path)
            file_data = response.content
            npy_data = np.load(BytesIO(file_data))
            return npy_data
        except Exception as e:
            print(f"Error downloading {dropbox_path}: {e}")
            if attempt + 1 == max_retries:
                print(f"Max retries reached for {dropbox_path}")
                return None
            else:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

def load_npy_from_dropbox(dbx, folder_path, max_workers=12):
    data_dict = {}

    try:
        folder_metadata = dbx.files_list_folder(folder_path)
        futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry in folder_metadata.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.npy'):
                    file_path = os.path.join(folder_path, entry.name)
                    subject_id = entry.name.replace('.npy', '')
                    future = executor.submit(download_npy_file, dbx, file_path)
                    futures[future] = subject_id
                    
            for future in as_completed(futures):
                subject_id = futures[future]
                npy = future.result()
                data_dict[subject_id] = npy

    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    return data_dict

def download_h5_file(dbx, dropbox_path, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Downloading {dropbox_path}, attempt {attempt + 1}")
            _, response = dbx.files_download(dropbox_path)
            file_data = response.content
            
            h5_data =h5py.File(BytesIO(file_data))

            return h5_data

        except Exception as e:
            print(f"Error downloading {dropbox_path}: {e}")
            if attempt + 1 == max_retries:
                print(f"Max retries reached for {dropbox_path}")
                return None
            else:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

def load_h5_from_dropbox(dbx, folder_path, max_workers=12):
    data_dict = {}
    try:
        folder_metadata = dbx.files_list_folder(folder_path)
        futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry in folder_metadata.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.h5'):
                    file_path = os.path.join(folder_path, entry.name)
                    numbers = re.findall(r'\d+', entry.name)
                    subject_id = numbers[-2]
                    future = executor.submit(download_h5_file, dbx, file_path)
                    futures[future] = subject_id

            for future in as_completed(futures):
                subject_id = futures[future]
                h5 = future.result()
                data_dict[subject_id] = h5

    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    return data_dict

# if __name__ == "__main__":
#     dbx = dropbox_connect()

#     npy_folder_path = ''
#     dict_npy = load_npy_from_dropbox(dbx, npy_folder_path)
#     print(f"Loaded {len(dict_npy)} npy files.")

#     h5_folder_path = ''
#     dict_h5 = load_h5_from_dropbox(dbx, h5_folder_path)
#     print(f"Loaded {len(dict_h5)} h5 files.")

#     from utils import ifft2c
#     dict_npy = {}
#     for subject_id, hf in dict_h5.items():
#         volume = hf['kspace'][()]
#         volume = rss_coil_combine(ifft2c(volume))
#         dict_npy[subject_id] = volume
#     print(len(dict_npy))