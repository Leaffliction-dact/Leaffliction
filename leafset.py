import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class LeafDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        arr = img.astype(np.float32) / 255.0
        # HWC -> CHW for arch purposes I guess?
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label
