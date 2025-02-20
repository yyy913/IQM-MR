from torch.utils.data import DataLoader
from dataloaders import IQA
from dataloaders.datasplit import split_train_test_csv

def gen_dataloader(cfg):
    split_train_test_csv(cfg)
    
    train_dataset = IQA.Train(cfg=cfg)
    
    if cfg.exclude:
        test_ref_dataset_in = IQA.Ref(cfg=cfg, subset='in')
        test_dataset_in = IQA.Test(cfg=cfg, subset='in')
        test_ref_dataset_ex = IQA.Ref(cfg=cfg, subset='ex')
        test_dataset_ex = IQA.Test(cfg=cfg, subset='ex')

        test_ref_loader_in = DataLoader(test_ref_dataset_in, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_loader_in = DataLoader(test_dataset_in, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_ref_loader_ex = DataLoader(test_ref_dataset_ex, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_loader_ex = DataLoader(test_dataset_ex, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

        return train_dataset, test_ref_loader_in, test_loader_in, test_ref_loader_ex, test_loader_ex
    
    else:
        test_ref_dataset = IQA.Ref(cfg=cfg)
        test_dataset = IQA.Test(cfg=cfg)

        test_ref_loader = DataLoader(test_ref_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

        return train_dataset, test_ref_loader, test_loader