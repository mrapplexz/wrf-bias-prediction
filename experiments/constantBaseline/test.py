import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './correction')
sys.path.insert(0, './experiments')

import torch
from torch.utils.data import DataLoader
from correction.config import cfg
from correction.models.loss import TurbulentMSE
import os

from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.scalers import StandardScaler
from correction.data.my_dataloader import WRFNPDataset
from correction.test import test_model
from correction.models.constantBias import ConstantBias

batch_size = int(input('input batch size: '))
run_id = int(input('input run id: '))
best_epoch = int(input('input epoch: '))
wrf_folder = './data/wrf/'

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
# logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

wrf_scaler = StandardScaler()
wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))

files = os.listdir(wrf_folder)
test_dataset = WRFNPDataset([wrf_folder + x for x in files], None)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))

model = ConstantBias().to(cfg.GLOBAL.DEVICE)
model.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.BASE_DIR, f'logs/constantBaseline/misc_{run_id}/models/model_{best_epoch}.pth')))
pytorch_total_params = sum(p.numel() for p in model.parameters())
# loss_values = [0. for _ in range(len(losses))]

save_dir = f'logs/constantBaseline/misc_{run_id}/'

if __name__ == '__main__':
    test_model(model, wrf_scaler, test_dataloader, files, './data/wrf_corr/')
