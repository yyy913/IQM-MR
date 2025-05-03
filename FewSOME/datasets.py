import sys
import os
from typing import Optional, Union
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mri_processing import crop, minmax_normalization


class SeveranceDataset(Dataset):
    def __init__(self,
                 df,            
                 indexes: Optional[Union[list, np.ndarray]] = None,
                 train: bool = True,
                 data_root: str = '/root/IQM-MR/data/severance',
                 resize: Optional[int] = None, ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.train = train
        self.data_root = data_root
        self.resize = resize
        self.indexes = indexes

        if indexes is not None and len(indexes) > 0:
            if train:                                
                self.df = self.df.iloc[indexes].reset_index(drop=True)
            else:                                      # validate / test
                complement = self.df.index.difference(indexes)
                df_sub     = self.df.loc[complement]

                n0, n1 = (df_sub['label'] == 0).sum(), (df_sub['label'] != 0).sum()

                if n0 > n1:           
                    keep0 = df_sub[df_sub['label'] == 0].iloc[:n1]
                    keep1 = df_sub[df_sub['label'] != 0]            
                else:                 
                    keep0 = df_sub[df_sub['label'] == 0]         
                    keep1 = df_sub[df_sub['label'] != 0].iloc[:n0]

                self.df = pd.concat([keep0, keep1]).reset_index(drop=True)

        self.targets = self.df['label'].to_numpy().astype(int)
        self.targets[self.targets != 0] = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int, seed: int = 1, base_ind: int = -1):
        base = False

        seq       = self.df.loc[idx, 'sequence']
        subject   = self.df.loc[idx, 'Subject ID']
        slice_idx = self.df.loc[idx, 'Slice Index']
        img_path  = os.path.join(self.data_root, seq, f'{subject}.npy')

        arr = np.load(img_path)[slice_idx]
        csize = min(arr.shape[-1], arr.shape[-2])
        arr = crop(arr, crop_size=csize)
        arr = minmax_normalization(arr)
        img = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        if self.resize is not None:
            img = F.interpolate(img,
                                size=(self.resize, self.resize),
                                mode='bilinear',
                                align_corners=False)

        img = img.squeeze(0)           
        img = img.repeat(3, 1, 1) 

        if self.train:
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes.tolist()) ) #if img2 is the same as img, regenerate ind
            c=1
            while (ind == idx):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes.tolist()) )
                c = c+1

            if ind == base_ind:
              base = True

            seq2       = self.df.loc[ind, 'sequence']
            subject2   = self.df.loc[ind, 'Subject ID']
            slice2     = self.df.loc[ind, 'Slice Index']
            img_path2  = os.path.join(self.data_root, seq2, f'{subject2}.npy')

            arr2 = np.load(img_path2)[slice2]
            arr2 = crop(arr2, crop_size=csize)
            arr2 = minmax_normalization(arr2)
            img2 = torch.from_numpy(arr2).float().unsqueeze(0).unsqueeze(0)
            if self.resize is not None:
                img2 = F.interpolate(img2,
                                     size=(self.resize, self.resize),
                                     mode='bilinear',
                                     align_corners=False)
            img2 = img2.squeeze(0).repeat(3, 1, 1)

            label = torch.FloatTensor([0])
        else:
            img2  = torch.Tensor([1])
            label = torch.FloatTensor([float(self.targets[idx])])

        return img, img2, label, base