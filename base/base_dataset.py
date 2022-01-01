from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class BaseDataset(Dataset):
    r"""
    Base Dataset Object

    Arguments:
        data (np.ndarray): data
        targets (np.ndarray): labels
    """
    idx: int  # requested data index
    x: torch.Tensor
    y: torch.Tensor

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
        raise NotImplementedError(f"Please verify whether dataset was properly initialized!")

