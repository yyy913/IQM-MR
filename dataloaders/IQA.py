import numpy as np

import torchvision
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from mri_processing import crop, minmax_normalization

from network.network_modules import PairGenerator
from dataloaders.get_df import get_df_v1

import warnings
warnings.simplefilter("ignore", UserWarning)

class Train(Dataset):
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset_name
        if cfg.dataset_name == 'Simulation':
            self.label_type = cfg.label_type
        else:
            self.label_type = 'label'
        self.image_dir, self.df = get_df_v1(cfg, is_train=True)
        if cfg.exclude:
            self.df = self.df[self.df['sequence'].isin(cfg.included_sequences)].copy()
        self.pg = PairGenerator(tau=cfg.tau)

    def __getitem__(self, idx):

        sample = dict()

        for im_num in range(self.im_num):
            sequence = self.df['sequence'].iloc[self.idx[im_num][idx]]
            subject_id = self.df['Subject ID'].iloc[self.idx[im_num][idx]]
            slice_idx = self.df['Slice Index'].iloc[self.idx[im_num][idx]]

            if self.dataset_name == 'Simulation':
                motion_level = self.df['Motion_Level'].iloc[self.idx[im_num][idx]]
                if motion_level == 0:
                    img_name = self.image_dir + f'{sequence}_clear/{subject_id}.npy'
                else:
                    img_name = self.image_dir + f'{sequence}_g{motion_level}/{subject_id}_motion.npy'
                img_label = self.df[self.label_type].iloc[self.idx[im_num][idx]]
                img = torch.from_numpy(np.load(img_name)).float()[slice_idx]

            elif self.dataset_name == 'Severance':
                img_name = self.image_dir + f'{sequence}/{subject_id}.npy'
                img_label = self.df['label'].iloc[self.idx[im_num][idx]]
                img = np.load(img_name)[slice_idx]
                img = torch.from_numpy(minmax_normalization(crop(img, crop_size=min(img.shape[-1], img.shape[-2])))).float()

            else:
                raise ValueError(f'Undefined database ({self.dataset_name}) has been given')

            sample[f'img_{im_num}_label'] = img_label
            sample[f'img_{im_num}_group'] = self.group[im_num][idx]
            sample[f'img_{im_num}'] = img

        return sample

    def __len__(self):
        return len(self.idx.transpose(1, 0))

    def get_pair_lists(self, batch_size, batch_list_len, im_num, uniform_select=False):
        self.im_num = im_num
        
        self.idx, self.group = self.pg.get_train_im_list(label=self.df[self.label_type].values, batch_size=batch_size, batch_list_len=batch_list_len, im_num=self.im_num, uniform_select=uniform_select)

class Test(Dataset):
    def __init__(self, cfg, subset='in'):
        self.dataset_name = cfg.dataset_name
        if cfg.dataset_name == 'Simulation':
            self.label_type = cfg.label_type
        else:
            self.label_type = 'label'
        _, _, self.image_dir, self.df_test = get_df_v1(cfg, is_train=False)

        if cfg.exclude:
            if subset == 'ex':
                self.df_test = self.df_test[self.df_test['sequence'] == cfg.exclude_sequence].copy()
            elif subset == 'in':
                self.df_test = self.df_test[self.df_test['sequence'].isin(cfg.included_sequences)].copy()

    def __getitem__(self, idx):
        sample = dict()

        sequence = self.df_test['sequence'].iloc[idx]
        subject_id = self.df_test['Subject ID'].iloc[idx]
        slice_idx = self.df_test['Slice Index'].iloc[idx]

        if self.dataset_name == 'Simulation':
            motion_level = self.df_test['Motion_Level'].iloc[idx]
            if motion_level == 0:
                img_name = self.image_dir + f'{sequence}_clear/{subject_id}.npy'
            else:
                img_name = self.image_dir + f'{sequence}_g{motion_level}/{subject_id}_motion.npy'
            img_label = self.df_test[self.label_type].iloc[idx]
            img = torch.from_numpy(np.load(img_name)).float()[slice_idx]

        elif self.dataset_name == 'Severance':
            img_name = self.image_dir + f'{sequence}/{subject_id}.npy'
            img_label = self.df_test['label'].iloc[idx]
            img = np.load(img_name)[slice_idx]
            img = torch.from_numpy(minmax_normalization(crop(img, crop_size=min(img.shape[-1], img.shape[-2])))).float()

        else:
            raise ValueError(f'Undefined database ({self.dataset_name}) has been given')

        sample['img_path'] = img_name + '_' + slice_idx
        sample['img_label'] = img_label
        sample[f'img'] = img            

        return sample

    def __len__(self):
        return len(self.df_test)

class Ref(Dataset):
    def __init__(self, cfg, subset='in'):
        self.dataset_name = cfg.dataset_name
        if cfg.dataset_name == 'Simulation':
            self.label_type = cfg.label_type
        else:
            self.label_type = 'label'
        self.image_dir, self.df_base, _, _ = get_df_v1(cfg, is_train=False)

        if cfg.exclude:
            if subset == 'ex':
                self.df_base = self.df_base[self.df_base['sequence'] == cfg.exclude_sequence].copy()
            elif subset == 'in':
                self.df_base = self.df_base[self.df_base['sequence'].isin(cfg.included_sequences)].copy()

        self.pg = PairGenerator(tau=cfg.tau)
        self.get_pair_lists(cfg.batch_size-1)

    def __getitem__(self, idx):
        sample = dict()

        sequence = self.df_base['sequence'].iloc[self.idx_0[idx]]
        subject_id = self.df_base['Subject ID'].iloc[self.idx_0[idx]]
        slice_idx = self.df_base['Slice Index'].iloc[self.idx_0[idx]]

        if self.dataset_name == 'Simulation':
            motion_level = self.df_base['Motion_Level'].iloc[self.idx_0[idx]]
            if motion_level == 0:
                img_name = self.image_dir + f'{sequence}_clear/{subject_id}.npy'
            else:
                img_name = self.image_dir + f'{sequence}_g{motion_level}/{subject_id}_motion.npy'
            img_label = self.df_base[self.label_type].iloc[self.idx_0[idx]]
            img = torch.from_numpy(np.load(img_name)).float()[slice_idx]


        elif self.dataset_name == 'Severance':
            img_name = self.image_dir + f'{sequence}/{subject_id}.npy'
            img_label = self.df_base['label'].iloc[self.idx_0[idx]]
            img = np.load(img_name)[slice_idx]
            img = torch.from_numpy(minmax_normalization(crop(img, crop_size=min(img.shape[-1], img.shape[-2])))).float()

        else:
            raise ValueError(f'Undefined database ({self.dataset_name}) has been given')

        sample['img_path'] = img_name + '_' + slice_idx
        sample['img_label'] = img_label
        sample[f'img'] = img 

        return sample

    def __len__(self):
        return len(self.idx_0)

    def get_pair_lists(self, batch_size):
        self.idx_0, self.group_0 = self.pg.get_test_im_list(label=self.df_base[self.label_type].values, batch_size=batch_size, random_choice=True)