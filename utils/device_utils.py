import torch


def get_device():
    """
    checks for the device type with torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def device_setup(n_gpu_in_use):
    """
    sets up gpu if available else cpu. In case of multiple gpu's returns indices for parrallel transfer
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_in_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_in_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_in_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_in_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_in_use > 0 else 'cpu')
    gpu_ids = list(range(n_gpu_in_use))
    return device, gpu_ids
