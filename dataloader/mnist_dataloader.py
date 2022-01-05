from base import BaseDataLoader
from torch.utils.data import DataLoader
from pathlib import Path

from dataloader.dataset import MNISTDataset
from utils import load_label_data, Path, load_image_data,ensure_dir


class MnistDataLoader(BaseDataLoader):
    r"""
    MNIST Data dataloader
    """

    def create_dataloader(
            self,
            root_path: str,
            data_file: str,
            label_file: str
    ) -> DataLoader:
        data_path = Path(f"{root_path}/{data_file}")
        label_path = Path(f"{root_path}/{label_file}")
        dir_exists = ensure_dir(root_path)
        if not dir_exists:
            print(f"Creating directory {root_path}. If you expected directory to exist please verify config. ")
        data = load_image_data(data_path)
        label_data = load_label_data(label_path)
        return DataLoader(
            dataset=MNISTDataset(data, label_data),
            **self.init_param_kwargs
        )
