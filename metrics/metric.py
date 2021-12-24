import torch


def accuracy(output, target):
    """ accuracy """
    with torch.no_grad():
        pred_ = torch.argmax(output, dim=1)
        assert pred_.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred_ == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """ accuracy of top-k values """
    with torch.no_grad():
        pred_ = torch.topk(output, k, dim=1, largest=True)[1]
        assert pred_.shape[0] == len(target)
        correct = 0
        for i in range(k):
            m_ = torch.tensor(pred_[:, i] == target)
            correct += torch.sum(m_).item()
    return correct / len(target)