import argparse
import collections

from config import ConfigParser
from task_test import TaskTest
from task_train import TaskTrain

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='ronbun')
    argparser.add_argument('-p', '--phase', default='train', type=str,
                      help='phase : either train or test')
    argparser.add_argument('-c', '--config', default='config/config.json', type=str,
                      help='config file path (default: None)')
    argparser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    argparser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    #args = argparser.parse_args()
    #for arg in vars(args):
    #   print(arg, getattr(args, arg))

    config = ConfigParser.from_args(argparser, options)

    if argparser.parse_args().phase == 'train':
        task = TaskTrain('train',config)
        task.build(config)
        task.train()
    else:
        task = TaskTest('test',config)
        task.build(config)
        task.eval()








