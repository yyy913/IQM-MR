import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import time

from haarPsi import haar_psi
from vsi import vsi
from vif import vif_p
from nqm import nqm

from mri_processing import *

def save_visualization(output_dir, key, slice_idx, ref_minmax, img_minmax, similarity_map, weight_map):
    """
    Helper function to save the visualization for a single slice.
    """
    plt.figure(figsize=(20, 8))
    
    # Reference image
    plt.subplot(2, 3, 1)
    plt.imshow(ref_minmax, cmap='gray', origin='lower')
    plt.title('Reference')
    plt.colorbar()
    
    # Distorted image
    plt.subplot(2, 3, 4)
    plt.imshow(img_minmax, cmap='gray', origin='lower')
    plt.title('Distorted')
    plt.colorbar()

    # Horizontal local similarity map
    plt.subplot(2, 3, 2)
    plt.imshow(similarity_map[:, :, 0], cmap='viridis', origin='lower')
    plt.title('Horizontal Local Similarity Map')
    plt.colorbar()

    # Vertical local similarity map
    plt.subplot(2, 3, 5)
    plt.imshow(similarity_map[:, :, 1], cmap='viridis', origin='lower')
    plt.title('Vertical Local Similarity Map')
    plt.colorbar()

    # Horizontal weight map
    plt.subplot(2, 3, 3)
    plt.imshow(weight_map[:, :, 0], cmap='magma', origin='lower')
    plt.title('Horizontal Weight Map')
    plt.colorbar()

    # Vertical weight map
    plt.subplot(2, 3, 6)
    plt.imshow(weight_map[:, :, 1], cmap='magma', origin='lower')
    plt.title('Vertical Weight Map')
    plt.colorbar()

    # Save the plot
    slice_dir = os.path.join(output_dir, key)
    os.makedirs(slice_dir, exist_ok=True)
    output_path = os.path.join(slice_dir, f'map_{slice_idx}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def calculate_iqm(clear_dict, motion_dict, folder_name):
    results = []
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    for key in tqdm(clear_dict.keys(), desc="Processing keys"):
        motion_key = f"{key}_motion" 
        if motion_key in motion_dict:
            # data_ref = rss_coil_combine(clear_dict[key])
            # data_img = rss_coil_combine(motion_dict[motion_key])
            data_ref = clear_dict[key]
            data_img = motion_dict[motion_key]
            crop_size = data_ref.shape[-1]
            data_ref = crop(data_ref, crop_size)
            data_img = crop(data_img, crop_size)

            for slice_idx in range(data_ref.shape[0]):
                ref_minmax = minmax_normalization(data_ref[slice_idx])
                img_minmax = minmax_normalization(data_img[slice_idx])
                Haarpsi, similarity_map, weight_map = haar_psi(ref_minmax * 255, img_minmax * 255)
                save_visualization(output_dir, key, slice_idx, ref_minmax*255, img_minmax*255, similarity_map, weight_map)
                x = torch.from_numpy(ref_minmax).unsqueeze(0).unsqueeze(1)
                y = torch.from_numpy(img_minmax).unsqueeze(0).unsqueeze(1)
                VSI = vsi(x, y)
                vif = vif_p(x, y)
                NQM = nqm(ref_minmax, img_minmax)
                results.append({
                    'Subject ID': key,
                    'Slice Index': slice_idx,
                    'Haarpsi': Haarpsi,
                    'VSI': VSI,
                    'VIF' : vif,
                    'NQM' : NQM,
                })

    df = pd.DataFrame(results)
    return df


def calculate_iqm_time(clear_dict, motion_dict):
    for key in clear_dict.keys():
        motion_key = f"{key}_motion"  
        if motion_key in motion_dict:
            # data_ref = rss_coil_combine(clear_dict[key])
            # data_img = rss_coil_combine(motion_dict[motion_key])
            data_ref = clear_dict[key]
            data_img = motion_dict[motion_key]
            crop_size = data_ref.shape[-1]
            data_ref = crop(data_ref, crop_size)
            data_img = crop(data_img, crop_size)

            for slice_idx in range(data_ref.shape[0]):
                ref_minmax = minmax_normalization(data_ref[slice_idx])
                img_minmax = minmax_normalization(data_img[slice_idx])
                x = torch.from_numpy(ref_minmax).unsqueeze(0).unsqueeze(1)
                y = torch.from_numpy(img_minmax).unsqueeze(0).unsqueeze(1)

                start_time = time.time()
                Haarpsi, similarity_map, weight_map = haar_psi(ref_minmax * 255, img_minmax * 255)
                haarpsi_time = time.time() - start_time

                start_time = time.time()
                VSI = vsi(x, y)
                vsi_time = time.time() - start_time

                start_time = time.time()
                vif = vif_p(x, y)
                vif_time = time.time() - start_time

                start_time = time.time()
                NQM = nqm(ref_minmax, img_minmax)
                nqm_time = time.time() - start_time

                print(f"Slice {slice_idx}: HaarPSI: {haarpsi_time:.6f}s, VSI: {vsi_time:.6f}s, VIF: {vif_time:.6f}s, NQM: {nqm_time:.6f}s")
            return