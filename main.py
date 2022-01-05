import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import MNISTConfig
from dataloader.mnist_dataloader import MnistDataLoader
from models import MnistModel
from task_runner.task_runner import TaskRunner
from tracker import TensorboardExperiment, Phase

# 1. initialize config store
cs = ConfigStore.instance()

# 2. load project config schema
cs.store(name="mnist_config", node=MNISTConfig)


# 3. set path and filename
@hydra.main(config_path="config/conf", config_name="config")
def main(config: MNISTConfig) -> None:

    OmegaConf.set_readonly(config, True)
    print(OmegaConf.to_yaml(config))

    # 4. define data-loaders

    data_loader = MnistDataLoader(
        config.params.batch_size,
        config.params.shuffle,
        config.params.num_workers
    )

    test_loader = data_loader.create_dataloader(
        root_path=config.paths.data,
        data_file=config.files.test_data,
        label_file=config.files.test_labels,
    )
    train_loader = data_loader.create_dataloader(
        root_path=config.paths.data,
        data_file=config.files.train_data,
        label_file=config.files.train_labels,
    )

    # 5. define model
    model = MnistModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.params.lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


    # 6. define task runners
    test_runner = TaskRunner(Phase.VAL, test_loader, model, loss_fn)
    train_runner = TaskRunner(Phase.TRAIN, train_loader, model, loss_fn, optimizer)

    # 7. define tracker and set log dir
    tracker = TensorboardExperiment(log_path=config.paths.log)

    tracker.add_graph(model, iter(test_runner.dataloader).next()[0])

    # 8. train
    for epoch_id in range(config.params.epoch_count):
        TaskRunner.run_epoch(test_runner, train_runner, tracker, epoch_id)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{config.params.epoch_count}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # reset
        train_runner.reset()
        test_runner.reset()

        # flush the tracker after every epoch for live updates
        tracker.flush()


if __name__ == "__main__":
    main()
