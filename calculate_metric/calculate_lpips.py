import lpips
from DISTS_pytorch import DISTS
import torch
import numpy as np
import os
import pandas as pd

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_lpips(modality):
    loss_fn = lpips.LPIPS(net='vgg').to(device)

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

def compute_dists(root="data", out_csv="dists.csv"):
    modalities = ["t1", "t1post", "t2", "flair"]
    groups     = [f"g{i}" for i in range(1, 6)]     # g1~g5
    model      = DISTS().to(device).eval()

    all_records = []

    for modality in modalities:
        dir_clear = os.path.join(root, f"{modality}_clear")

        for group in groups:
            dir_motion = os.path.join(root, f"{modality}_{group}")

            clear_files = [fn for fn in os.listdir(dir_clear) if fn.endswith(".npy")]

            for fn_clear in clear_files:
                path0 = os.path.join(dir_clear,  fn_clear)
                fn_motion = fn_clear.replace(".npy", "_motion.npy")
                path1 = os.path.join(dir_motion, fn_motion)

                im0 = np.load(path0)
                im1 = np.load(path1)

                im0 = (im0 - im0.min()) / (im0.max() - im0.min())
                im1 = (im1 - im1.min()) / (im1.max() - im1.min())

                im0 = np.repeat(im0[:, None, :, :], 3, axis=1)
                im1 = np.repeat(im1[:, None, :, :], 3, axis=1)

                im0_t = torch.tensor(im0, dtype=torch.float32, device=device)
                im1_t = torch.tensor(im1, dtype=torch.float32, device=device)

                with torch.no_grad():
                    dists = model(im0_t, im1_t).cpu().numpy() 

                subject_id = os.path.splitext(fn_clear)[0]        
                for s, d_val in enumerate(dists):
                    all_records.append(
                        [subject_id, s, modality, group, float(d_val)]
                    )

    df = pd.DataFrame(
        all_records,
        columns=["Subject ID", "Slice Index", "sequence", "Motion_Level", "DISTS"]
    )
    df.to_csv(out_csv, index=False)