import sys
sys.path.insert(0, '../')
import torch
import os
from correction.config import cfg
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def test(dataloader, model, criterion, wrf_scaler, era_scaler, logger):
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        for test_data, test_label in tqdm(dataloader):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)
            output = model(test_data)
            output = wrf_scaler.channel_inverse_transform(output)
            test_data = wrf_scaler.channel_inverse_transform(test_data)
            test_label = era_scaler.channel_inverse_transform(test_label)
            loss = criterion(test_data, output, test_label, logger)
            test_loss += loss.item()
        test_loss = test_loss / len(dataloader)
        if logger:
            logger.print_stat_readable()
    return test_loss


def test_model(model, loss, wrf_scaler, era_scaler, dataloader, logs_dir, logger=None):
    test_losses = 0
    i = 0
    with torch.no_grad():
        model.eval()
        for test_data, test_label in tqdm(dataloader):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)
            output = model(test_data)
            # todo правильный подсчет метрики относительного улучшения качества
            test_losses += loss(test_data, output, test_label, logger)

        test_losses = test_losses / len(dataloader)

    return test_losses
