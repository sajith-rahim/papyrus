import numpy as np
import torch

from base import BaseDataset


class MNISTDataset(BaseDataset):
    idx: int  # requested data index
    x: torch.Tensor
    y: torch.Tensor

    # From Pytorch docs.
    MAX_VAL = 255.0
    MEAN = 0.1306604762738429
    STDEV = 0.3081078038564622

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        if len(data) != len(targets):
            raise ValueError(
                "data and targets must be the same length. "
                f"{len(data)} != {len(targets)}"
            )

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.get_x(idx)
        y = self.get_y(idx)
        return x, y

    def get_x(self, idx: int):
        self.idx = idx
        self.preprocess_x()
        return self.x

    def preprocess_x(self):
        self.x = self.data[self.idx].copy().astype(np.float64)
        self.x /= self.MAX_VAL
        self.x -= self.MEAN
        self.x /= self.STDEV
        self.x = self.x.astype(np.float32)
        self.x = torch.from_numpy(self.x)
        self.x = self.x.unsqueeze(0)

    def get_y(self, idx: int):
        self.idx = idx
        self.preprocess_y()
        return self.y

    def preprocess_y(self):
        self.y = self.targets[self.idx]
        self.y = torch.tensor(self.y, dtype=torch.long)
