#!/bin/bash

# Update pip to the latest version
pip install --upgrade pip

# Install core libraries
pip install dropbox numpy scipy pandas matplotlib tqdm ipywidgets h5py
pip install seaborn
pip install wandb
pip install einops
pip install timm

# Install medical imaging libraries
pip install SimpleITK nibabel

# Install PyTorch (specific version may vary based on your system)
pip install torch torchvision torchaudio
pip install argparse wandb tensorboard

# Install scikit
pip install scikit-image
pip install scikit-learn

# Install image quality assessment library (piq)
pip install piq
pip install lpips
pip install pyrtools
pip install PyWavelets
pip install opencv-python

pip install tensorflow

# google API
pip install google-api-python-client
pip install oauth2client
pip install httplib2