import datetime
import os
import torch

from utils import ensure_dir


class Checkpointer:

    def __init__(self, task_name):
        self.task_name = task_name
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.iteration = None

    def generate_checkpoint_path(self, path, iteration, metric_val):
        if path is None:
            path = os.getcwd()
        ensure_dir(path)
        epoch_metric_val = f"{iteration}-{metric_val}"[:7]
        ts_identifier = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        filename = f"{self.task_name}-{ts_identifier}-{epoch_metric_val}.pth.zip"
        filepath = os.path.join(path, filename)
        return filepath

    def save_checkpoint(self, path , iteration, metric_val, model, optimizer):
        r"""
        Save checkpoint.
        """
        checkpoint_path = self.generate_checkpoint_path(path, iteration, metric_val)
        if type(model) == torch.nn.DataParallel:
            # converting a DataParallel model to be able to load later without DataParallel
            self.model_state_dict = model.module.state_dict()
        else:
            self.model_state_dict = model.state_dict()

        self.optimizer_state_dict = optimizer.state_dict()
        self.iteration = iteration

        torch.save(self, checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint
