import random
from datetime import datetime

import numpy as np
import torch

from Consts.consts import Experiment_suffix, use_cuda


def log_f(msg):
    log_name = f'DATA/Logs/log-{Experiment_suffix}.txt'
    sttime = datetime.now().strftime('%m-%d_%H:%M:%S - ')
    with open(log_name, 'a') as fp:
        fp.write(sttime + msg + '\n')


def print_cuda_memory_status(suffix):
    if torch.cuda.is_available():
        to_gb = 1024 * 1024 * 1024
        total_memory = torch.cuda.get_device_properties(0).total_memory / to_gb
        c = torch.cuda.memory_cached(0) / to_gb
        a = torch.cuda.memory_allocated(0) / to_gb
        f = total_memory - c - a
        print(torch.cuda.get_device_name(0))
        print(f'Memory Status Total: {total_memory:.4f}, \t Allocated: {a:.4f}'
              f', \t Cached: {c:.4f}, \t Free: {f:.4f} \t - {suffix}')


def get_dev():
    if use_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        # print('device used: cuda')
    else:
        dev = torch.device("cpu")
        # print('device used: cpu')
    return dev


def init_seeds():
    seed = 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
