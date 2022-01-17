import random
import torch
import numpy as np
import argparse


def set_all_seeds(random_seed):
    """
    Sets random seeds in numpy, torch, and random.

    Args:
        random_seed: The seed to set. 
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    return

def dict2namespace(config):
    """
    Converts a given dictionary to an argparse namespce object.

    Args:
        config: The dictionary to convert to namespace.
                Can contain up to one level of nested dicts.
    
    Returns:
        namespace: The converted namespace.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
