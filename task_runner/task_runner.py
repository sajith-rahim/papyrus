import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from metrics import Metric
from store import Checkpointer
from tracker import Phase, ExperimentTracker
from utils import get_device


class TaskRunner:
    def __init__(
            self,
            phase,
            dataloader: DataLoader,
            model: torch.nn.Module,
            loss_fn,
            checkpoint_cfg,
            optimizer=None,
    ) -> None:
        self.phase = phase
        self.dataloader = dataloader
        self.model = model
        if phase == Phase.TRAIN and optimizer is None:
            raise AttributeError("No optimizer defined!")
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # checkpointer
        self.checkpointer = Checkpointer(model.__class__.__name__)
        self.checkpoint_cfg = checkpoint_cfg
        # * * *
        self.run_count = 0
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        self.model.train(self.phase is Phase.TRAIN)
        device = get_device()
        for x, y in tqdm(self.dataloader, desc=desc, ncols=120):
            x = x.to(device)
            y = y.to(device)
            loss, batch_accuracy = self._run_single_batch(x, y)
            experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)

            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _run_single_batch(self, x, y):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x)
        loss = self.loss_fn(prediction, y)

        # Compute Batch Validation Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []

    def set_checkpoint(self, epoch_id, metric_val):
        interval = self.checkpoint_cfg.save_interval
        if not isinstance(interval,int):
            raise AttributeError("Invalid interval value. check config [checkpoint.save_interval]")
        if epoch_id % interval == 0:
            self.checkpointer.save_checkpoint(
                self.checkpoint_cfg.path,
                epoch_id,
                metric_val,
                self.model,
                self.optimizer
            )

    @staticmethod
    def run_epoch(
            test_runner,
            train_runner,
            experiment: ExperimentTracker,
            epoch_id: int,
    ):
        # Training Loop
        experiment.set_phase(Phase.TRAIN)
        train_runner.run("Train Progress", experiment)

        # Log Training Epoch Metrics
        experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)

        # Testing Loop
        experiment.set_phase(Phase.VAL)
        test_runner.run("Validation Progress:", experiment)

        # Save checkpoint
        train_runner.set_checkpoint(epoch_id, train_runner.avg_accuracy)

        # Log Validation Epoch Metrics
        experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch_id)
        experiment.add_epoch_confusion_matrix(
            test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id
        )
