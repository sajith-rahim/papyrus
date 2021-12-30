import torch
import numpy as np
import dataloaders.dataloader as dataloader
import models.model as model
from trainer.trainer import Trainer
from utils import device_setup
import metrics.losses as losses
import metrics.metric as metric

class TaskTrain:

    def __init__(self, taskname, config, seed = 1001):
        phase = 'train'
        self.config = config
        self.taskname = taskname
        self.n_gpu = config['n_gpu']
        self.logger = config.get_logger(phase)
        self.setseed(seed)

    def setseed(self, seed_value):
        SEED = seed_value
        torch.manual_seed(SEED)
        # cuDNN to only use deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        # disable cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)

    def build(self,config):

        self.data_loader = config.init_obj('data_loader', dataloader)
        self.validation_data_loader = self.data_loader.split_validation()

        # build model architecture, then print to console
        self.model = config.init_obj('arch', model)
        self.logger.info(self.model)

        # loss + metrics
        self.loss = getattr(losses, config['loss'])
        self.metrics = [getattr(metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, self.trainable_params)
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)


    def train(self):

        device, device_ids = device_setup(self.n_gpu)
        self.model = self.model.to(device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        trainer = Trainer(self.model, self.loss, self.metrics, self.optimizer,
                          config=self.config,
                          device=device,
                          data_loader=self.data_loader,
                          valid_data_loader=self.validation_data_loader,
                          lr_scheduler=self.lr_scheduler)

        trainer.train()

