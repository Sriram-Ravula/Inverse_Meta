import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

from datasets.celeba import CelebA
from datasets.ffhq import FFHQ 

def get_dataset(config):    
    if config.data.dataset == 'celeba':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()]), 
                                 download=True)
        else:
            dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor()]), 
                                 download=True)

        test_dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor()]), 
                                  download=True)
    
    elif config.data.dataset == "ffhq":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(config.data.data_path), 
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()]), 
                                resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(config.data.data_path), 
                            transform=transforms.ToTensor(),
                            resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    else:
        raise NotImplementedError("Dataset not supported!")

    return dataset, test_dataset

def split_dataset(base_dataset, hparams):
    """
    Split a given dataset into train, val, and test sets.
    If we do not want a validation set, returns None for val.

    Args:
        base_dataset: The dataset to use for splitting.
                      Type: Dataset.
        hparams: The experiment parameters to use for splitting.
                 Type: Namespace.
                 Expected to have the constituents:
                    hparams.data.num_train - int
                    hparams.data.num_val - int
                    hparams.data.num_test - int
                    hparams.outer.use_validation - bool
                    hparams.seed - int
    
    Returns:
        datasets: A dict containing the train, val, and test datasets.
                  Type: dict.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    use_validation = hparams.outer.use_validation

    indices = list(range(len(base_dataset)))

    random_state = np.random.get_state()
    np.random.seed(hparams.seed+1)
    np.random.shuffle(indices)
    np.random.set_state(random_state)

    if use_validation:
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train+num_val]
        test_indices = indices[num_train+num_val:num_train+num_val+num_test]

        train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(base_dataset, test_indices)
    else:
        train_indices = indices[:num_train]
        test_indices = indices[num_train:num_train+num_test]

        train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
        val_dataset = None
        test_dataset = torch.utils.data.Subset(base_dataset, test_indices)
    
    out_dict = {'train': train_dataset, 
            'val': val_dataset, 
            'test': test_dataset}

    return out_dict