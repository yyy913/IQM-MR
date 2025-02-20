from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from httplib2 import Http
from oauth2client import file, client, tools
import io
from io import BytesIO
import os
import re
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def google_drive_connect():
    # 인증 파일 경로 설정
    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
    store = file.Storage('storage.json')
    creds = store.get()

    # 명령줄 인자 우회 처리
    try:
        import argparse
        flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args([])
    except ImportError:
        flags = None

    # 인증 처리
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store, flags)

    # Google Drive API 서비스 객체 생성
    service = build('drive', 'v3', http=creds.authorize(Http()))
    
    return service


def download_npy_file(service, file_id, file_name, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Downloading {file_name}, attempt {attempt + 1}")
            request = service.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            npy_data = np.load(fh, allow_pickle=False)  # set allow_pickle to False for security if not needed
            return npy_data
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            if attempt + 1 == max_retries:
                print(f"Max retries reached for {file_name}")
                return None
            else:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

def list_files_in_folder(service, folder_id):
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
            pageSize=100,  # 요청당 최대 100개
            pageToken=page_token
        ).execute()
        files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if not page_token:
            break
    return files

def load_npy_from_google_drive(service, folder_id, max_workers=1):
    data_dict = {}
    try:
        files = list_files_in_folder(service, folder_id)
        print(len(files))
        futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file in files:
                if file['name'].endswith('.npy'):
                    file_name = file['name']
                    subject_id = file_name.replace('.npy', '')
                    future = executor.submit(download_npy_file, service, file['id'], file['name'])
                    futures[future] = subject_id

            for future in as_completed(futures):
                subject_id = futures[future]
                npy = future.result()
                if npy is not None:
                    data_dict[subject_id] = npy
                    print(subject_id, npy.shape)

    except Exception as e:
        print(f"Error accessing folder {folder_id}: {e}")

    return data_dict


# if __name__ == "__main__":
    # folder_id = "13EQdbzv5ckrE6N1QUJrpxmt6hJMoQ-5L" 
    # service = google_drive_connect()
    # t1_clear = load_npy_from_google_drive(service, folder_id, max_workers=1)