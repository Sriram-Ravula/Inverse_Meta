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
    if config.data.dataset == 'Brain-Multicoil':
        train_files = get_all_files(config.data.train_input_dir, pattern='*T2*.h5')
        test_files = get_all_files(config.data.test_input_dir, pattern='*T2*.h5')

        dataset = BrainMultiCoil(train_files,
                                input_dir=config.data.train_input_dir,
                                maps_dir=config.data.train_maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

        test_dataset = BrainMultiCoil(test_files,
                                input_dir=config.data.test_input_dir,
                                maps_dir=config.data.test_maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

    elif config.data.dataset == 'Knee-Multicoil':
        train_files = get_all_files(config.data.train_input_dir, pattern='*.h5')
        test_files = get_all_files(config.data.test_input_dir, pattern='*.h5')

        dataset = KneesMultiCoil(train_files,
                                input_dir=config.data.train_input_dir,
                                maps_dir=config.data.train_maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

        test_dataset = KneesMultiCoil(test_files,
                                input_dir=config.data.test_input_dir,
                                maps_dir=config.data.test_maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)

    elif config.data.dataset == 'Knees-Singlecoil':
        train_files = get_all_files(config.data.train_input_dir, pattern='*.h5')
        test_files = get_all_files(config.data.test_input_dir, pattern='*.h5')

        dataset = KneesSingleCoil(train_files,
                                image_size = config.data.image_size,
                                R=config.data.R,
                                pattern=config.data.pattern,
                                orientation=config.data.orientation)

        test_dataset = KneesSingleCoil(test_files,
                                image_size = config.data.image_size,
                                R=config.data.R,
                                pattern=config.data.pattern,
                                orientation=config.data.orientation)

    else:
        raise NotImplementedError("Dataset not supported!")

    return dataset, test_dataset

def split_dataset(train_set, test_set, hparams):
    """
    Split a given dataset into train, val, and test sets.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    tr_indices = list(range(len(train_set)))
    te_indices = list(range(len(test_set)))

    print("Train Dataset Size: ", len(train_set))
    print("Test Dataset Size: ", len(test_set))

    random_state = np.random.get_state()
    np.random.seed(hparams.seed)
    np.random.shuffle(tr_indices)
    np.random.seed(hparams.seed)
    np.random.shuffle(te_indices)
    np.random.set_state(random_state)

    train_indices = tr_indices[:num_train]
    val_indices = tr_indices[num_train:num_train+num_val]
    test_indices = te_indices[:num_test]

    train_dataset = torch.utils.data.Subset(train_set, train_indices)
    val_dataset = torch.utils.data.Subset(train_set, val_indices)
    test_dataset = torch.utils.data.Subset(test_set, test_indices)

    out_dict = {'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}
    

    return out_dict
