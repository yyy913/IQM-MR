import lpips
import torch
import numpy as np
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='vgg').to(device)

def compute_lpips(modality):

    dir0 = f"data/{modality}_clear"
    groups = [f"g{i}" for i in range(1, 6)]  # g1 ~ g5

    for group in groups:
        dir1 = f"data/{modality}_{group}"
        output_csv = f"{modality}_{group}_lpips.csv"
        results = []
        
        for filename in os.listdir(dir0):
            if filename.endswith(".npy") and filename in os.listdir(dir1):
                path0 = os.path.join(dir0, filename)
                path1 = os.path.join(dir1, filename)

                im0 = np.load(path0)  
                im1 = np.load(path1)  

                im0 = (im0 - im0.min()) / (im0.max() - im0.min()) * 2 - 1
                im1 = (im1 - im1.min()) / (im1.max() - im1.min()) * 2 - 1

                im0 = np.repeat(im0[:, np.newaxis, :, :], 3, axis=1)
                im1 = np.repeat(im1[:, np.newaxis, :, :], 3, axis=1)

                im0 = torch.tensor(im0, dtype=torch.float32).to(device)
                im1 = torch.tensor(im1, dtype=torch.float32).to(device)

                for s in range(im0.shape[0]):  
                    d = loss_fn(im0[s:s+1], im1[s:s+1])  
                    score = d.item()
                    results.append([filename, s + 1, score])

        df = pd.DataFrame(results, columns=["Filename", "Slice", "LPIPS_Score"])
        df.to_csv(output_csv, index=False)