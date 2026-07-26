import os

import torch
import numpy as np

from preprocessor import add_features, modified_zscore


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_type='pos'):
        self.data = data
        if data_type == 'pos':
            self.label = np.ones(self.data.shape[0])
        else:
            self.label = np.zeros(self.data.shape[0])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.label[index]
        return X, Y


class LazyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, data_type='pos', norm=False,
                 cut=1500, length=3000, tile=3, feature_window_size=3):

        self.npy_file = npy_path
        self.data = np.load(self.npy_file, allow_pickle=True)
        self.data_type = data_type
        self.norm = norm
        self.param_cut = cut
        self.param_length = length
        self.param_tile = tile
        self.feature_window_size = feature_window_size

        self.index_map = []  # save (read_idx, tile_idx, segment_start)
        step = length // tile

        for read_idx, signal in enumerate(self.data):
            if len(signal) < cut + length:
                continue
            signal = signal[cut:]
            for tile_idx in range(tile):
                start = tile_idx * step
                end = start + length
                while end <= len(signal):
                    self.index_map.append((read_idx, end - length))
                    end += length

        self.total_sample_num = len(self.index_map)
        if data_type == 'pos':
            self.label = np.ones(self.total_sample_num, dtype=np.int64)
        else:
            self.label = np.zeros(self.total_sample_num, dtype=np.int64)

    def __len__(self):
        return self.total_sample_num

    def __getitem__(self, index):
        read_idx, start_pos = self.index_map[index]
        signal = self.data[read_idx]
        signal = signal[self.param_cut:]
        
        # segment = modified_zscore(signal[start_pos:start_pos + self.param_length]) if self.norm else signal[start_pos:start_pos + self.param_length]
        segment = signal[start_pos:start_pos + self.param_length]
        X = add_features(
            np.array([segment]),
            norm=self.norm,
            s_len=self.param_length,
            w_len=self.feature_window_size,
        )[0].float()
        Y = self.label[index]
        # print(X.shape, Y.shape)
        return X, Y
