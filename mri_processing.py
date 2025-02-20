import numpy as np
import SimpleITK as sitk
import torch

def ifft2c(k):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c(img):
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, (-2,-1)), norm='ortho'), (-2,-1))
    return k

# def ifft2c(k):
#     x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
#     return x

# def fft2c(img):
#     k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, (-2,-1)), norm='ortho'), (-2,-1))
#     return k

def rss_coil_combine(image):
    squared_abs = np.abs(image)**2
    sum_of_squares = np.sum(squared_abs, axis=1)
    combined_images = np.sqrt(sum_of_squares)
    return combined_images

def Znormalization(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        tmp = image - mean/(std+1e-17)
    else:
        tmp = (image - mean) / (std)
    return tmp

def minmax_normalization(image):
    min_val = np.min(image)
    max_val = np.max(image)
    tmp = (image - min_val) / (max_val - min_val)
    return tmp

def crop(img, crop_size=320):
    height, width = img.shape[-2:]
    w_from = max(0, (width - crop_size) // 2)
    w_to = min(width, w_from + crop_size)
    h_from = (height - crop_size) // 2
    h_to = h_from + crop_size

    return img[:, h_from:h_to, w_from:w_to]

def padding(volume, size=320):
    slices, height, width = volume.shape
    pad_width = max(0, size - width) // 2 

    padded_volume = np.pad(
        volume,
        pad_width=((0, 0), (0, 0), (pad_width, pad_width)), 
        mode='edge'
    )
    return padded_volume

def correct_bias_field(input_np):
    input_sitk = sitk.GetImageFromArray(input_np)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    mask_sitk = sitk.OtsuThreshold(input_sitk, 0, 1, 200)
    corrected_image = corrector.Execute(input_sitk, mask_sitk)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_sitk)
    corrected_image_full_resolution = input_sitk / sitk.Exp(log_bias_field)
    # corrected_np = sitk.GetArrayFromImage(corrected_image)
    corrected_np_full_resolution = sitk.GetArrayFromImage(corrected_image_full_resolution)
    return corrected_np_full_resolution