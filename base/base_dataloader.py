from torch import randperm
from torch._utils import _accumulate
from torch.utils.data import DataLoader, Subset


class BaseDataLoader(DataLoader):
    r"""
    Base DataLoader Object

    Arguments:
        dataset (Dataset): Dataset
        batch_size (int): batch size
        shuffle (bool): Shuffle data
        num_workers(int): number of workers
    """

    def __init__(self, batch_size, shuffle, num_workers):
        self.init_param_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

    def create_dataloader(self,
            root_path: str,
            data_file: str,
            label_file: str,
    ) -> DataLoader:
        raise NotImplementedError(f"Please verify whether dataloader was properly initialized!")

    @staticmethod
    def random_split(dataset, lengths):
        r"""
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        Arguments:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths of splits to be produced
        """
        if sum(lengths) != len(dataset):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        indices = randperm(sum(lengths)).tolist()
        return [Subset(dataset, indices[offset - length:offset]) for offset, length in
                zip(_accumulate(lengths), lengths)]
