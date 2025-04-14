import os
import sys
import math

from scipy.stats import spearmanr, pearsonr
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network.optimizer_utils import get_optimizer, get_scheduler
from network.Regressor import Regressor
from dataloaders import dataloader_gen

from utils.util import load_model, save_model
from utils.util import set_wandb, write_log


def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    if cfg.exclude:
        train_dataset, _, test_loader_in, _, test_loader_ex = dataloader_gen.gen_dataloader(cfg)
    else:
        train_dataset, _, test_loader = dataloader_gen.gen_dataloader(cfg)

    if cfg.dataset_name == 'Simulation':
        train_dataset_scores = train_dataset.df[cfg.label_type].values
    elif cfg.dataset_name == 'Severance':
        train_dataset_scores = train_dataset.df['label'].values

    cfg.n_scores = len(np.unique(train_dataset_scores))

    cfg.log_file = cfg.log_configs()

    write_log(cfg.log_file, f'[*] {cfg.n_scores} scores exist.')

    model = Regressor(cfg)

    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = model.cuda()

    optimizer = get_optimizer(cfg, model)
    lr_scheduler = get_scheduler(cfg, optimizer)

    if cfg.load:
        load_model(cfg, model, optimizer=optimizer, load_optim_params=False)

    if cfg.test_first:
        model.eval()
        if cfg.exclude:
            write_log(cfg.log_file, '\n[*] Internal Test')
            srcc, pcc, mae = evaluation(cfg, model, test_loader_in)
            sys.stdout.write(f'IN - [SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')
            write_log(cfg.log_file, '\n[*] External Test')
            srcc, pcc, mae = evaluation(cfg, model, test_loader_ex)
            sys.stdout.write(f'EX - [SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')
        else:
            srcc, pcc, mae = evaluation(cfg, model, test_loader)
            sys.stdout.write(f'[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

    best_srcc = 0.85
    best_pcc = 0.85

    best_srcc_total_results = []
    best_pcc_total_results = []

    best_srcc_epoch = -1
    best_pcc_epoch = -1

    log_dict = dict()
    for epoch in range(0, cfg.epoch):
        model.train()
        model.backbone.encoder.eval()

        if (epoch + 1) <= 5:
            uniform_select = True
        else:
            uniform_select = False

        train_dataset.get_pair_lists(batch_size=cfg.batch_size, batch_list_len=cfg.im_list_len, im_num=cfg.im_num, uniform_select=uniform_select)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)

        mae_loss = train(cfg, model, optimizer, train_loader, epoch)
        write_log(cfg.log_file, f'\nEpoch: {(epoch + 1):d} MAE Loss: {mae_loss:.3f}\n')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['LR'] = lr_scheduler.get_lr()[0] if lr_scheduler else cfg.lr

        if ((epoch + 1) == 1) | (((epoch + 1) >= cfg.start_eval) & ((epoch + 1) % cfg.eval_freq == 0)):

            model.eval()
            if cfg.exclude:
                write_log(cfg.log_file, '\n[*] Internal Test')
                srcc, pcc, mae = evaluation(cfg, model, test_loader_in)
                sys.stdout.write(f'IN - [SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')
                write_log(cfg.log_file, '\n[*] External Test')
                srcc, pcc, mae = evaluation(cfg, model, test_loader_ex)
                sys.stdout.write(f'EX - [SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')
            else:
                srcc, pcc, mae = evaluation(cfg, model, test_loader)
                sys.stdout.write(f'\n[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

            if srcc > best_srcc:
                best_srcc = srcc
                best_srcc_epoch = epoch
                best_srcc_total_results = [srcc, pcc, mae]
                save_model(cfg, model, optimizer, epoch, [srcc, pcc, mae], criterion='SRCC')

            if pcc > best_pcc:
                best_pcc = pcc
                best_pcc_epoch = epoch
                best_pcc_total_results = [srcc, pcc, mae]
                save_model(cfg, model, optimizer, epoch, [srcc, pcc, mae], criterion='PCC')

            if cfg.wandb:
                log_dict['Test/SRCC'] = srcc
                log_dict['Test/PCC'] = pcc
                log_dict['Test/MAE'] = mae

        if cfg.wandb:
            wandb.log(log_dict)

        print('lr: %.6f' % (optimizer.param_groups[0]['lr']))
        if lr_scheduler:
            lr_scheduler.step()

    write_log(cfg.log_file, 'Training End')
    write_log(cfg.log_file,
              'Best SRCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_srcc_epoch, best_srcc_total_results[0], best_srcc_total_results[1], best_srcc_total_results[2]))
    write_log(cfg.log_file,
              'Best PCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_pcc_epoch, best_pcc_total_results[0], best_pcc_total_results[1], best_pcc_total_results[2]))
    print(cfg.save_folder)

def train(cfg, model, optimizer, data_loader, epoch):
    loss_fn = nn.L1Loss()
    mae_losses = 0

    dataloader_iterator = iter(data_loader)

    for idx in range(len(dataloader_iterator) // cfg.batch_size):

        optimizer.zero_grad()

        predictions = []
        scores = []
        groups = []
        for dl_iter in range(cfg.batch_size):
            sample = next(dataloader_iterator)

            scores_tmp = torch.cat([sample[f'img_{im_idx}_label'] for im_idx in range(cfg.im_num)])
            scores_tmp = scores_tmp.cuda().float()
            scores.append(scores_tmp)

            groups_tmp = torch.cat([sample[f'img_{im_idx}_group'] for im_idx in range(cfg.im_num)])
            groups_tmp = groups_tmp.cuda()
            groups.append(groups_tmp)

            for im_idx in range(cfg.im_num):
                img = sample[f'img_{im_idx}'].cuda()
                img = img.unsqueeze(1)
                if img.shape[1] == 1:
                    img = img.repeat(1, 3, 1, 1)
                output = model(img).squeeze()
                predictions.append(output)

        predictions = torch.stack(predictions)
        scores = torch.cat(scores)
        groups = torch.cat(groups)

        loss = loss_fn(predictions, scores)

        loss.backward()
        optimizer.step()

        mae_losses += loss.item()

        sys.stdout.write(
            f'\r[Epoch {epoch + 1}/{cfg.epoch}] [Batch {idx + 1}/{cfg.im_list_len}] [Loss {loss.item():.2f}]')

    return mae_losses / (idx + 1)

def evaluation(cfg, model, data_loader):
    model.eval()
    if cfg.dataset_name == 'Simulation':
        test_label_gt = data_loader.dataset.df_test[cfg.label_type].values
    elif cfg.dataset_name == 'Severance':
        test_label_gt = data_loader.dataset.df_test['label'].values

    preds_list = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            for idx, sample in enumerate(data_loader):

                if idx % 1 == 0:
                    sys.stdout.write(f'\rExtract Test Img Features... [{idx + 1}/{len(data_loader)}]')

                image = sample[f'img'].cuda()
                image = image.unsqueeze(1)

                if image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)

                output = model(image)
                preds_list.append(output)

    preds_np = torch.stack(preds_list).cpu().detach().numpy().squeeze()
    print(preds_np.shape)

    srcc = spearmanr(preds_np, test_label_gt)[0]
    pcc = pearsonr(preds_np, test_label_gt)[0]
    mae = np.abs(preds_np - test_label_gt).mean()

    write_log(cfg.log_file, f'\nTest MAE: {mae: .4f} SRCC: {srcc: .4f} PCC: {pcc: .4f}')

    return srcc, pcc, mae

if __name__ == "__main__":
    from configs.config_v1 import ConfigV1 as Config

    cfg = Config()
    main(cfg)