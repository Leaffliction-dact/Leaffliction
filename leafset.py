import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class LeafDataset(Dataset):
    def __init__(self, samples, size=None, mean=None, std=None):
        self.samples = samples
        self.size = size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        if (self.size is not None):
            img = cv2.resize(img, (self.size, self.size))
        if (self.mean is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        if (self.mean is not None):
            arr = (arr - np.array(self.mean)) / np.array(self.std)
        # HWC -> CHW for arch purposes I guess?
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor, label
