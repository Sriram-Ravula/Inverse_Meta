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

def split_dataset(base_dataset, hparams):
    """
    Split a given dataset into train, val, and test.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    indices = list(range(len(base_dataset)))

    random_state = np.random.get_state()
    np.random.seed(hparams.seed+1)
    np.random.shuffle(indices)
    np.random.set_state(random_state)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:num_train+num_val+num_test]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(base_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def init_c(hparams):
    c_type = hparams.outer.hyperparam_type
    m = hparams.problem.num_measurements

    if c_type == 'scalar':
        c = torch.tensor(1.)
    elif c_type == 'vector':
        c = torch.ones(m)
    elif c_type == 'matrix':
        c = torch.eye(m)
    else:
        raise NotImplementedError 
    
    return c

def get_meta_optimizer(opt_params, hparams):
    lr = hparams.outer.lr
    lr_decay = hparams.outer.lr_decay
    opt_type = hparams.outer.optimizer

    if opt_type == 'adam':
        meta_opt = torch.optim.Adam([{'params': opt_params}], lr=lr)
    elif opt_type == 'sgd':
        meta_opt = torch.optim.SGD([{'params': opt_params}], lr=lr)
    else:
        raise NotImplementedError
    
    if lr_decay:
        meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, lr_decay)
        return (meta_opt, meta_scheduler)
    else:
        return meta_opt
