import random
import torch
import torch.fft as torch_fft
import numpy as np
import argparse
import os
import yaml
import torch.utils.tensorboard as tb
import time
from datetime import datetime
import sigpy as sp
import torch.nn as nn



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
    if hparams.problem.measurement_type == 'mri':
        if c_type == 'scalar':
            c = torch.tensor(1.)
        elif c_type == 'vector':
            c = torch.ones(hparams.problem.measurement_shape)
        elif c_type == 'matrix':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
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

def parse_config(config_path):
    with open(config_path, 'r') as f:
        hparams = yaml.safe_load(f)

    if hparams['use_gpu']:
        num = hparams['gpu_num']
        hparams['device'] = torch.device('cuda:'+str(num)) if torch.cuda.is_available() else torch.device('cpu:0')
    else:
        hparams['device'] = torch.device('cpu:0')

    if hparams['net']['model'] != 'ncsnv2':
        raise NotImplementedError
    if hparams['outer']['meta_type'] not in ['implicit', 'maml', 'mle']:
        raise NotImplementedError

    if hparams['data']['dataset'] == "celeba":
        hparams['data']['image_size'] = 64
    elif hparams['data']['dataset'] == "ffhq":
        hparams['data']['image_size'] = 256
    elif hparams['data']['dataset'] == "mri":
        hparams['data']['image_size'] = 384
    else:
        raise NotImplementedError

    hparams['data']['image_shape'] = (hparams['data']['num_channels'], hparams['data']['image_size'], hparams['data']['image_size'])
    hparams['data']['n_input'] = np.prod(hparams['data']['image_shape'])

    #automatically set ROI to eye region if not specified
    if hparams['outer']['ROI'] and not isinstance(hparams['outer']['ROI'], tuple):
        if hparams['data']['dataset'] == "celeba":
            hparams['outer']['ROI'] = ((27, 15),(35, 35))
        elif hparams['data']['dataset'] == "ffhq":
            hparams['outer']['ROI'] = ((90, 50),(60, 156))
        else:
            raise NotImplementedError

    #TODO implement finite difference
    if hparams['outer']['finite_difference'] or hparams['outer']['measurement_loss'] \
        or hparams['outer']['meta_loss_type'] != 'l2':
        raise NotImplementedError

    if hparams['problem']['measurement_type'] == 'circulant':
        hparams['problem']['train_indices'] = np.random.randint(0, hparams['data']['n_input'], hparams['problem']['num_measurements'])
        hparams['problem']['sign_pattern'] = np.float32((np.random.rand(1, hparams['data']['n_input']) < 0.5)*2 - 1.)
    elif hparams['problem']['measurement_type'] == 'superres':
        hparams['problem']['y_shape'] = (hparams['data']['num_channels'], hparams['data']['image_size']//hparams['problem']['downsample_factor'], hparams['data']['image_size']//hparams['problem']['downsample_factor'])
        hparams['problem']['num_measurements'] = np.prod(hparams['problem']['y_shape'])
    elif hparams['problem']['measurement_type'] == 'identity':
        hparams['problem']['y_shape'] = hparams['data']['image_shape']
        hparams['problem']['num_measurements'] = hparams['data']['n_input']
    elif hparams['problem']['measurement_type'] == 'inpaint':
        hparams['problem']['num_measurements'] = hparams['data']['n_input'] - hparams['data']['num_channels']*hparams['problem']['inpaint_size']**2
    elif hparams['problem']['measurement_type'] == 'mri':
        hparams['problem']['measurement_shape'] = (16,
                                                   hparams['data']['image_size'],
                                                   hparams['data']['image_size'])


    if hparams['problem']['add_noise'] or hparams['problem']['add_dependent_noise']:
        raise NotImplementedError

    HPARAMS = dict2namespace(hparams)

    return HPARAMS

# computes mvue from kspace and coil sensitivities
def get_mvue(kspace, s_maps):
    '''
    Get mvue estimate from coil measurements

    Parameters:
    -----------
    kspace : complex np.array of size b x c x n x n
            kspace measurements
    s_maps : complex np.array of size b x c x n x n
            coil sensitivities

    Returns:
    -------
    mvue : complex np.array of shape b x n x n
            returns minimum variance estimate of the scan
    '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=-3) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))

# Multicoil forward operator for MRI
class MulticoilForwardMRI(nn.Module):
    def __init__(self, orientation):
        self.orientation = orientation
        super(MulticoilForwardMRI, self).__init__()
        return

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x


    '''
    Inputs:
     - image = [B, H, W] torch.complex64/128    in image domain
     - maps  = [B, C, H, W] torch.complex64/128 in image domain
     - mask  = [B, W] torch.complex64/128 w/    binary values
    Outputs:
     - ksp_coils = [B, C, H, W] torch.complex64/128 in kspace domain
    '''
    def forward(self, image, maps, mask):
        # Broadcast pointwise multiply
        coils = image[:, None] * maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        if self.orientation == 'vertical':
            # Mask k-space phase encode lines
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == 'horizontal':
            # Mask k-space frequency encode lines
            ksp_coils = ksp_coils * mask[:, None, :, None]
        else:
            if len(mask.shape) == 3:
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError('mask orientation not supported')


        # Return downsampled k-space
        return ksp_coils

def parse_args(docstring):
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--doc', type=str, default=now, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--verbose', type=str, default='low', help='Verbose level: low | med | high')

    args = parser.parse_args()

    return args

