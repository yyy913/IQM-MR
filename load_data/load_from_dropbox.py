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

def list_all_files(dbx, folder_path):
    all_entries = []
    try:
        result = dbx.files_list_folder(folder_path)
        all_entries.extend(result.entries)
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            all_entries.extend(result.entries)
    except Exception as e:
        print("Error retrieving file list:", e)
    return all_entries

def download_and_save_file(dbx, dropbox_path, local_path, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt+1} for downloading {dropbox_path}")
            _, response = dbx.files_download(dropbox_path)
            file_data = response.content
            with open(local_path, 'wb') as f:
                f.write(file_data)
            return True
        except Exception as e:
            print(f"Download failed for {dropbox_path}: {e}")
            if attempt + 1 == max_retries:
                print(f"Maximum retry attempts reached for {dropbox_path}")
                return False
            else:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

def download_and_save_all_files(dbx, folder_path, local_dir, max_workers=12):
    all_entries = list_all_files(dbx, folder_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in all_entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.npy'):
                dropbox_file_path = os.path.join(folder_path, entry.name)
                local_file_path = os.path.join(local_dir, entry.name)
                future = executor.submit(download_and_save_file, dbx, dropbox_file_path, local_file_path)
                futures[future] = entry.name
        
        for future in as_completed(futures):
            file_name = futures[future]
            result = future.result()
            if not result:
                print(f"Failed to save {file_name}")

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

def upload_folder_to_dropbox(local_folder, dropbox_folder):
    dbx = dropbox_connect()

    for root, dirs, files in os.walk(local_folder):
        relative_path = os.path.relpath(root, local_folder)
        dropbox_path = os.path.join(dropbox_folder, relative_path).replace("\\", "/")
        
        if relative_path != ".":
            try:
                dbx.files_create_folder(dropbox_path)
            except dropbox.exceptions.ApiError as e:
                if e.error.is_path() and isinstance(e.error.get_path().is_conflict(), dropbox.files.FolderConflictError):
                    pass  

        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            dropbox_file_path = os.path.join(dropbox_path, file_name).replace("\\", "/")
            
            with open(local_file_path, "rb") as f:
                print(f"Uploading {file_name} to {dropbox_file_path}")
                dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

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