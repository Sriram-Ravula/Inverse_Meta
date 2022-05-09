from dataclasses import replace
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import glob

from datasets.mri_dataloaders import BrainMultiCoil, KneesSingleCoil, KneesMultiCoil

def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

def get_dataset(config):
    num_train = config.data.num_train
    num_val = config.data.num_val
    num_test = config.data.num_test
    total_samples = num_train + num_val + num_test

    if config.data.dataset == 'Brain-Multicoil':
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        files = np.random.choice(files, size=total_samples, replace=False)

        dataset = None
        test_dataset = BrainMultiCoil(files,
                                input_dir=config.data.input_dir,
                                maps_dir=config.data.maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

    elif config.data.dataset == 'Knee-Multicoil':
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        files = np.random.choice(files, size=total_samples, replace=False)

        dataset = None
        test_dataset = KneesMultiCoil(files,
                                input_dir=config.data.input_dir,
                                maps_dir=config.data.maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

    elif config.data.dataset == 'Knees-Singlecoil':
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        files = np.random.choice(files, size=total_samples, replace=False)

        dataset = None
        test_dataset = KneesSingleCoil(files,
                                image_size = config.data.image_size,
                                R=config.data.R,
                                pattern=config.data.pattern,
                                orientation=config.data.orientation)

    else:
        raise NotImplementedError("Dataset not supported!")

    return dataset, test_dataset

def split_dataset(base_dataset, hparams):
    """
    Split a given dataset into train, val, and test sets.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    indices = list(range(len(base_dataset)))

    print("Dataset Size: ", len(base_dataset))

    random_state = np.random.get_state()
    np.random.seed(hparams.seed)
    np.random.shuffle(indices)
    np.random.set_state(random_state)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:num_train+num_val+num_test]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(base_dataset, test_indices)

    out_dict = {'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}

    return out_dict
