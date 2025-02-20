import pandas as pd

def get_df_v1(cfg, is_train=False):
    if cfg.dataset_name == 'Simulation':
        if is_train:
            train_image_path = cfg.dataset_root
            train_df_path = cfg.datasplit_root + 'fr-iqm_train.csv'

        else:
            ref_image_path = cfg.dataset_root
            ref_df_path = cfg.datasplit_root + 'fr-iqm_train.csv'

            test_image_path = cfg.dataset_root
            test_df_path = cfg.datasplit_root + 'fr-iqm_test.csv'

    elif cfg.dataset_name == 'Severance':
        if is_train:
            pass

        else:
            pass
    else:
        raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')

    if is_train:
        return train_image_path, pd.read_csv(train_df_path)
    elif is_train is False:
        return ref_image_path, pd.read_csv(ref_df_path), test_image_path, pd.read_csv(test_df_path)
    else:
        raise ValueError(f'Undefined mode ({is_train}) has been given')