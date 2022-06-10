# -*- coding:utf-8 -*-

"""
@File    : time_series_dataset.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/01/23 17:29
@Function: 
"""
import torch
import numpy as np


def create_tensor_dataset(df):
    if type(df) is np.ndarray:
        sequences = df.tolist()
    else:
        sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    dataset = torch.stack(dataset)
    n_seq, seq_len, n_features = dataset.shape
    return dataset, seq_len, n_features


class TimeSeries_Dataset(torch.utils.data.Dataset):
    def __init__(self, ts_data, labels):
        self.ts_data = ts_data
        self.labels = labels

    def __len__(self):
        return self.ts_data.shape[0]

    def __getitem__(self, index):
        return self.ts_data[index], self.labels[index]