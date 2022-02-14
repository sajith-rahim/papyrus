from itertools import islice
from pathlib import Path

import torch

import hydra

from dataloader.dataset import MNISTDataset
from models import MnistModel
from store import Checkpointer
from utils import load_image_data, load_label_data, get_device


@hydra.main(config_path="config/conf", config_name="config")
def infer(config):
    root_path = config.paths.data
    data_file = config.files.test_data
    label_file = config.files.test_labels
    label_path = Path(f"{root_path}/{label_file}")
    data_path = Path(f"{root_path}/{data_file}")
    data = load_image_data(data_path)
    label_data = load_label_data(label_path)

    # TODO: Load Image data
    mnist = MNISTDataset(data, label_data)
    itr = iter(mnist)
    sample, label = next(islice(itr, 115, 116))
    sample = sample.unsqueeze(0).to(get_device())

    model = MnistModel().to(get_device())
    checkpoint = Checkpointer.load_checkpoint(config.checkpoint.checkpoint_id, config.checkpoint.path,
                                              str(get_device()))
    model.load_state_dict(checkpoint.model_state_dict)

    prediction = model(sample)

    print(f"Prediction={torch.argmax(prediction)} : True Label={label}")


if __name__ == "__main__":
    infer()
