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
