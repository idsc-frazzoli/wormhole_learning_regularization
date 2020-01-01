from __future__ import print_function, division, absolute_import
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


class Signal(Dataset):
    def __init__(self, npy_file, transform=None, train=True, test=False):
        # npy_file: path to the npy file where all the paths are indicated
        # transform: optional transform to be applied on a sample

        with open(npy_file, 'r') as f:
            s = f.read()
            path_all = s.split(' ')
        path_num = len(path_all)

        self.transform = transform

        self.path_all = path_all

    def __len__(self):
        return len(self.path_all)

    def __getitem__(self, idx):
        path = self.path_all[idx]
        label = int(path.split('_')[-1].split('.')[0])
        data = np.load(path).astype(np.float32)
        data = self.transform(data)
        return data, label
