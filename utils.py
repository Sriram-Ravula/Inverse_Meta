import random
import torch
import numpy as np


def set_all_seeds(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    return