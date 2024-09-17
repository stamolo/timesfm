import torch
import torch.nn as  nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, datas, patch_len):
        assert datas.shape[1] % patch_len == 0
        self.datas = datas
        self.patch_len = patch_len
        
        self.eps = 1.0e-8

    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, idx):
        datas = self.datas[idx]

        inputs = datas[:-self.patch_len]
        targets = datas[self.patch_len:]
        x = torch.tensor(inputs, dtype=torch.float32) # (TP,)
        y = torch.tensor(targets, dtype=torch.float32) # (TP,)

        mean = torch.mean(x).detach()
        std = torch.sqrt(torch.var(x, unbiased=True) + self.eps).detach()

        # normalize
        x = (x - mean) / std
        y = (y - mean) / std

        return x, y

if __name__ == "__main__":
    import glob
    import numpy as np
    import pandas as pd

    data_folder = './datas/'
    paths = glob.glob(data_folder + '*.csv')

    patch_len = 32
    seq_len = 16
    batch = 64

    x = np.arange(0, 4000)
    xs = []

    xs = np.array(xs)
    print(xs.shape)

    datas = []
    for path in paths:
        data = pd.read_csv(path)
        data = np.asarray(data['Data'])
        for i in range(len(data) - (patch_len*(seq_len+1))):
            datas.append(data[i:i+patch_len*(seq_len+1)])

    datas = np.array(datas)
    print(datas.shape)

    dataset = SeqDataset(datas, patch_len)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print('Input:', x.shape, 'Target:', y.shape)
