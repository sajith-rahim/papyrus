import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from tracker.phase import Phase
from utils import create_log_dir
from pathlib import Path


class TensorboardExperiment:
    def __init__(self, log_path: str, create: bool = True):
        self.phase = Phase.TRAIN
        self._writer = SummaryWriter(
            log_dir=create_log_dir(log_path, parents=True)
        )
        plt.ioff()

    def set_phase(self, phase: Phase):
        self.phase = phase

    def flush(self):
        self._writer.flush()

    def add_graph(self,model, input):
        self._writer.add_graph(model,input, verbose=False)

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.phase}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.phase}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_confusion_matrix(
            self, y_true: list, y_pred: list, step: int
    ):
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        fig = self.create_confusion_matrix(y_true, y_pred, step)
        tag = f"{self.phase}/epoch/confusion_matrix"
        self._writer.add_figure(tag, fig, step)

    @staticmethod
    def collapse_batches(
            y_true: list, y_pred: list
    ) -> tuple:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_confusion_matrix(
            self, y_true: list, y_pred: list, step: int
    ) -> plt.Figure:
        cm = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues")
        cm.ax_.set_title(f"{self.phase} Epoch: {step}")
        return cm.figure_
