from tqdm import tqdm

import dataloaders._dataloader as dataloader
import models.model as model_
import metrics.losses as losses
import metrics.metric as metric
import torch


class TaskTest:

    def __init__(self, taskname, config):
        phase = 'test'
        self.config = config
        self.taskname = taskname
        self.n_gpu = config['n_gpu']
        self.logger = config.get_logger(phase)

    def load_weights(self, resume_point):
        self.logger.info('Loading checkpoint: {} ...'.format(resume_point))
        checkpoint = torch.load(resume_point)
        state_dict = checkpoint['state_dict']
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.logger.info('Loaded checkpoint: {} ...'.format(resume_point))

    def build(self, config):
        self.data_loader = getattr(dataloader, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        model = config.init_obj('arch', model_)
        self.logger.info(model)

        # get function handles of loss and metrics
        self.loss = getattr(losses, config['loss'])
        self.metrics = [getattr(metric, met) for met in config['metrics']]

        self.load_weights(self, config.resume)

    def eval(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metrics))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(device), target.to(device)
                output = self.model(data)

                # self.track_output(output)

                # computing loss, metrics
                loss = self.loss(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metrics):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metrics)
        })
        self.logger.info(log)

    def track_output(self):
        pass
