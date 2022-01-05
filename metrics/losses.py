import torch.nn.functional as F


def nll_loss(output, target):
    """ negative log likelihhod loss """
    return F.nll_loss(output, target)


def binary_cross_entropy(output, target):
    """ binary cross entropy loss """
    return F.binary_cross_entropy(output, target)