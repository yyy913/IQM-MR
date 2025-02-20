import numpy as np

import numpy as np

class PairGenerator:
    def __init__(self, tau=0.1):
        self.tau = tau

    def get_train_im_list(self, fr_iqm, batch_size, batch_list_len=150, im_num=3, uniform_select=False):

        fr_iqm = np.array(fr_iqm)

        sample_idx = [[] for _ in range(im_num)]
        sample_group = [[] for _ in range(im_num)]

        fr_iqm_unique, fr_iqm_num = np.unique(fr_iqm, return_counts=True)

        if uniform_select:
            groups = np.array_split(fr_iqm_unique, min(len(fr_iqm_unique), batch_size))
        else:
            groups = np.array_split(fr_iqm_unique, min(len(fr_iqm_unique), batch_size * 3))

        batch_list = batch_list_len

        for _ in range(batch_list):

            if len(groups) >= batch_size:
                groups_selected = np.random.choice(np.arange(len(groups)), batch_size, replace=False)
                groups_selected = [groups[gs_idx] for gs_idx in groups_selected]

            else:
                groups_selected = groups

            for g, group in enumerate(groups_selected):

                for im_idx in range(im_num):
                    random_score = np.random.choice(group)
                    random_idx = np.random.choice(np.where(fr_iqm == random_score)[0])

                    sample_idx[im_idx].append(random_idx)
                    sample_group[im_idx].append(g)

        sample_idx = np.array(sample_idx)
        sample_group = np.array(sample_group)

        return sample_idx, sample_group

    def get_test_im_list(self, fr_iqm, batch_size, iteration=1, random_choice=False):

        fr_iqm = np.array(fr_iqm)

        sample_idx = []
        sample_group = []

        fr_iqm_unique, fr_iqm_num = np.unique(fr_iqm, return_counts=True)
        groups = np.array_split(fr_iqm_unique, min(len(fr_iqm_unique), batch_size))

        for _ in range(iteration):

            for g, group in enumerate(groups):

                if random_choice:
                    random_score = np.random.choice(group)
                    random_idx_0 = np.random.choice(np.where(fr_iqm == random_score)[0])
                else:
                    random_score = group[-1]
                    random_idx_0 = np.where(fr_iqm == random_score)[0][0]

                sample_idx.append(random_idx_0)
                sample_group.append(g)

        sample_idx_0 = np.array(sample_idx)
        sample_group_0 = np.array(sample_group)

        return sample_idx_0, sample_group_0