import torch
import numpy as np
np.random.seed(42)

from timesfm.timesfm import TimesFM, TimesFMConfig
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./datas/(x-c)xsin(x).csv')
parser.add_argument('--load_weights_folder', type=str, default='./weights/24/')
parser.add_argument('--start_point', type=int, default=1800)
parser.add_argument('--forecast_len', type=int, default=500)
args = parser.parse_args()

data = args.data
load_weights_folder = args.load_weights_folder
start_t = args.start_point
forecast_len = args.forecast_len

config = TimesFMConfig()

device = config.device
patch_len = config.patch_len
seq_len = config.seq_len

df = pd.read_csv(data)
data = np.asarray(df['Data'])
        
timesfm = TimesFM(config).to(device)
timesfm.load_state_dict(torch.load(load_weights_folder + 'timesfm.pth', weights_only=True))
timesfm.eval()

data = data[start_t-(patch_len*seq_len)+1:start_t+1]
fcst_point = len(data)

x = data

fcst = []
for i in range(math.ceil(forecast_len / patch_len)):
    y = timesfm.infer(x)
    fcst.append(y)
    x = np.concatenate((x, y), axis=0)
    x = x[patch_len:]
    
tmp = np.empty_like(data)
tmp[:] = np.nan
fcst = np.concatenate(fcst, axis=0)
fcst = fcst[:forecast_len]
fcst = np.concatenate((tmp, fcst), axis=0)

data = np.asarray(df['Data'])
data = data[start_t-(patch_len*seq_len)+1:start_t+forecast_len]

plt.plot(data, color='blue', label='ground truth')
plt.plot(fcst, color='red', label='forecast')
plt.axvline(x=fcst_point, color='black')
plt.legend()

plt.show()