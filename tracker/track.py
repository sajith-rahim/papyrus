from tracker.phase import Phase

class ExperimentTracker():
    def set_phase(self, phase: Phase):
        """Sets the current phase of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(
            self, y_true: list, y_pred: list, step: int
    ):
        """Implements logging a confusion matrix at epoch-level."""
