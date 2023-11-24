import copy
import sys

sys.path.insert(0, '../')
import torch
import os
from correction.config import cfg
from tqdm import tqdm
import numpy as np


def test_model(model, wrf_scaler, dataloader, files, out_dir):
    with torch.no_grad():
        model.eval()
        stack = []
        for test_data in (pbar := tqdm(dataloader)):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)

            test_data = wrf_scaler.channel_transform(test_data, 2)

            output = model(test_data)

            output = wrf_scaler.channel_inverse_transform(output, 2)
            test_data = wrf_scaler.channel_inverse_transform(test_data, 2)
            stack.extend(output.transpose(0, 1))
        out_stack = []
        for i, item in enumerate(stack):
            is_last = i == len(stack) - 1
            if not is_last:
                out_stack.append(item[0])
            else:
                out_stack.extend(item)
        out_stack_split = []
        tmp = []
        for t in out_stack:
            tmp.append(t)
            if len(tmp) == 24:
                out_stack_split.append(torch.stack(tmp).cpu().numpy())
                tmp = []
        for file, item in zip(files, out_stack_split):
            np.save(out_dir + '/corrected_' + file, item)


def calculate_metric(wrf_orig, wrf_corr, era, criterion):
    loss_orig = criterion(wrf_orig, wrf_orig, era)
    loss_corr = criterion(wrf_orig, wrf_corr, era)
    print(loss_orig, loss_corr)
    metric = max((loss_orig - loss_corr) / loss_orig, 0)
    return metric
