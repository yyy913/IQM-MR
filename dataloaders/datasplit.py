import pandas as pd
import numpy as np
import os

def bin_iqm(df, bin):
    df_copy = df.copy()
    label_types = ['Haarpsi', 'VSI', 'VIF', 'NQM', 'LPIPS']

    for column_name in label_types:
        data = df_copy[column_name]
        counts, bin_edges = np.histogram(data, bins=bin, range=(data.min(), data.max()))
        bin_indices = np.digitize(data, bins=bin_edges[:-1], right=False) - 1
        bin_indices = np.clip(bin_indices, 0, bin - 1)
        if column_name == 'LPIPS':
            bin_indices = (bin - 1) - bin_indices
        df_copy[column_name] = bin_indices
    
    unique_keys = df_copy[['Subject ID', 'Slice Index', 'sequence']].drop_duplicates()
    unique_keys['Motion_Level'] = 0
    unique_keys['Haarpsi'] = bin
    unique_keys['VSI'] = bin
    unique_keys['VIF'] = bin
    unique_keys['NQM'] = bin
    unique_keys['LPIPS'] = bin

    df_result = pd.concat([df_copy, unique_keys], ignore_index=True)

    return df_result

def select_subject_ids(group, n=20, random_state=42):
    unique_subj = group["Subject ID"].unique()
    sample_size = min(n, len(unique_subj))  
    chosen_subj = np.random.choice(unique_subj, sample_size, replace=False)
    return group[group["Subject ID"].isin(chosen_subj)]

def split_train_test_csv(cfg):
    if cfg.dataset_name == 'Simulation':
        df = bin_iqm(pd.read_csv(cfg.label_df_root), cfg.bin)

        df_test = (
            df.groupby("sequence", group_keys=False)
            .apply(lambda g: select_subject_ids(g, n=20, random_state=42))
            .reset_index(drop=True)
        )
        df_train = df.drop(df_test.index)

        train_df_path = cfg.datasplit_root + 'fr-iqm_train.csv'
        test_df_path = cfg.datasplit_root + 'fr-iqm_test.csv'

        if not os.path.exists(cfg.datasplit_root):
            os.makedirs(cfg.datasplit_root)

        df_train.to_csv(train_df_path, index=False)
        df_test.to_csv(test_df_path, index=False)