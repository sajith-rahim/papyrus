from argparse import Namespace

from torch.utils.data import DataLoader

TRAIN_CONFIG = Namespace(split_name='Training', shuffle=True, drop_last=True)
VAL_CONFIG = Namespace(split_name='Validation', shuffle=False, drop_last=False)

"""

config-
    |_data
        |_train
            |_dataset
            |_config
                |_data_dir(string)
                |_data_list(string)
                |_transforms[
                                |_transform
                                |_config
                            ]
"""


def create_data_loader(input_config, loader_config, pin_memory=True):
    dataset = input_config.dataset(input_config.config)
    print(f'{loader_config.split_name} set: {len(dataset)} examples')
    data_loader = DataLoader(
        dataset,
        input_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=loader_config.drop_last
    )
    return data_loader
