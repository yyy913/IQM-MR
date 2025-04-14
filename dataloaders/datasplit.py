import pandas as pd
import numpy as np
import os
import random

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

def stratified_split_sequence(group, test_ratio, random_state=42):
    random.seed(random_state)
    train_indices = []
    test_indices = []
    
    for label in range(6):
        label_rows = group[group['label'] == label]
        indices = label_rows.index.tolist()
        
        if len(indices) == 0:
            raise ValueError(f"Label {label} not found in the group. Ensure that each label is present in the group.")
        if len(indices) == 1:
            raise ValueError(f"Only one sample found for label {label}. Cannot split into train and test.")
        else:
            n_test = max(1, int(len(indices) * test_ratio))
            if n_test == len(indices):
                n_test = len(indices) - 1

            random.shuffle(indices)
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
    
    train_group = group.loc[train_indices]
    test_group = group.loc[test_indices]
    return train_group, test_group

def cap_label_zero(group, cap=200, random_state=42):
    zero_mask = group['label'] == 0
    zero_group = group[zero_mask]
    non_zero_group = group[~zero_mask]
    if len(zero_group) > cap:
        zero_group = zero_group.sample(n=cap, random_state=random_state)
    return pd.concat([non_zero_group, zero_group])

def split_data(df, cfg):
    df = df.groupby('sequence').apply(lambda grp: cap_label_zero(grp, cap=200, random_state=42)).reset_index(drop=True)
    
    train_list = []
    test_list = []
    
    if hasattr(cfg, 'exclude') and cfg.exclude == True:
        for seq, group in df.groupby('sequence'):
            if seq in cfg.exclude_sequences:
                test_ratio = 0.8
            else:
                test_ratio = 0.2
            train_group, test_group = stratified_split_sequence(group, test_ratio, random_state=42)
            train_list.append(train_group)
            test_list.append(test_group)
    else:
        for seq, group in df.groupby('sequence'):
            train_group, test_group = stratified_split_sequence(group, 0.2, random_state=42)
            train_list.append(train_group)
            test_list.append(test_group)
    
    df_train = pd.concat(train_list).reset_index(drop=True)
    df_test = pd.concat(test_list).reset_index(drop=True)
    return df_train, df_test

def split_train_test_csv(cfg):
    if cfg.dataset_name == 'Simulation':
        df = bin_iqm(pd.read_csv(cfg.label_df_root), cfg.bin)
        if cfg.exclude == True:
            df_excluded = df[df["sequence"] == cfg.exclude_sequence]
            df_excluded_selected = (
                df_excluded.groupby("sequence", group_keys=False)
                .apply(lambda g: select_subject_ids(g, n=20, random_state=42))
                .reset_index(drop=True)
            )
            df_excluded_remaining = df_excluded.drop(df_excluded_selected.index)

            df_rest = df[df["sequence"] != cfg.exclude_sequence]
            df_test = (
                df_rest.groupby("sequence", group_keys=False)
                .apply(lambda g: select_subject_ids(g, n=20, random_state=42))
                .reset_index(drop=True)
            )
            df_test = pd.concat([df_test, df_excluded_remaining], ignore_index=True)

        else:
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

    elif cfg.dataset_name == 'Severance':
        df = pd.read_csv(cfg.label_df_root)
        df_train, df_test = split_data(df, cfg)

        train_df_path = cfg.datasplit_root + 'severance_train.csv'
        test_df_path = cfg.datasplit_root + 'severance_test.csv'

        if not os.path.exists(cfg.datasplit_root):
            os.makedirs(cfg.datasplit_root)

        df_train.to_csv(train_df_path, index=False)
        df_test.to_csv(test_df_path, index=False)

    else:
        raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')