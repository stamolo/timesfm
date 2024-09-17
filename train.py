import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import progress_bar
import numpy as np
np.random.seed(42)

from datasets.dataset import SeqDataset
from timesfm.timesfm import TimesFM, TimesFMConfig
from torch.utils.data import DataLoader
import os
import glob
import pandas as pd

config = TimesFMConfig()

save_weights_folder = './weights/'
batch = 64
epochs = 200
lr = 2.5e-4

device = config.device
patch_len = config.patch_len
seq_len = config.seq_len

data_folder = './datas/'
paths = glob.glob(data_folder + '*.csv')

datas = []
for path in paths:
    data = pd.read_csv(path)
    data = np.asarray(data['Data'])
    for i in range(len(data) - (patch_len*(seq_len+1))):
        datas.append(data[i:i+patch_len*(seq_len+1)])

datas = np.array(datas)

dataset = SeqDataset(datas, patch_len)
        
timesfm = TimesFM(config).to(device)

optimizer = torch.optim.Adam(timesfm.parameters(), lr=lr)

def run_epoch():
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch)

    losses = []
    for i, (x, t) in enumerate(dataloader):
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        y = timesfm(x)

        loss = F.mse_loss(y, t)
        
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch} loss: {loss:.4f} lr: {lr:e} ', end='')
        progress_bar(i, len(dataloader), bar_size=30)
        losses.append(loss.item())
    return sum(losses) / len(losses)

patience = 0
for epoch in range(1, epochs+1):
    timesfm.train()
    loss = run_epoch()

    print(f'mean loss: {loss:.4f}')
    if epoch > 1:
        if loss > prev_loss:
            patience += 1
            print(f'patience: {patience:3d}')
            if patience >= 5:
                break
        else:
            if not os.path.exists(save_weights_folder + str(epoch)):
                os.makedirs(save_weights_folder + str(epoch))
            torch.save(timesfm.state_dict(), save_weights_folder + str(epoch) + '/timesfm.pth')
            prev_loss = loss
    else:
        prev_loss = loss
        if not os.path.exists(save_weights_folder + str(epoch)):
            os.makedirs(save_weights_folder + str(epoch))
        torch.save(timesfm.state_dict(), save_weights_folder + str(epoch) + '/timesfm.pth')