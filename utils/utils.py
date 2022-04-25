import random
import torch
import torch.fft as torch_fft
import numpy as np
import argparse
import os
import yaml
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.nn as nn
import scipy as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from utils.loss_utils import get_measurements, get_transpose_measurements

plt.rcParams["savefig.bbox"] = 'tight'

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

    return train_dataset, val_dataset, test_dataset

def init_c(hparams):
    c_type = hparams.outer.hyperparam_type
    init_val = float(hparams.outer.hyperparam_init)

    if hparams.problem.measurement_type == 'mri':
        if c_type == 'scalar':
            c = torch.tensor(1.)
        elif c_type == 'vector':
            c = torch.ones(hparams.problem.measurement_shape) * init_val
        elif c_type == 'matrix':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        m = hparams.problem.num_measurements

        if c_type == 'scalar':
            c = torch.tensor(init_val)
        elif c_type == 'vector':
            c = torch.ones(m) * init_val
        elif c_type == 'matrix':
            c = torch.eye(m) * init_val
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
        hparams['problem']['measurement_shape'] = (hparams['data']['image_size'], hparams['data']['image_size'])


    if hparams['problem']['add_dependent_noise']:
        raise NotImplementedError

    HPARAMS = dict2namespace(hparams)

    print(yaml.dump(HPARAMS, default_flow_style=False))

    return HPARAMS

def parse_args(docstring, manual=False, config=None, doc=None):
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--doc', type=str, default=now, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')

    if manual:
        args = parser.parse_args(["--config", config, "--doc", doc])
    else:
        args = parser.parse_args()

    return args


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

def plot_images(images, title, save=False, fname=None):
    #TODO THIS NEEDS FIXING!
    """Function to plot and/or save an image"""

    fig = plt.figure(figsize=(1, 1))

    grid_img = torchvision.utils.make_grid(images.cpu(), nrow=images.shape[0]//2).permute(1, 2, 0)

    ax = fig.add_subplot(1, 1, 1, frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(grid_img)

    if save:
        plt.savefig(fname)
    else:
        ax.set_title(title)
        plt.show();

def get_measurement_images(images, hparams):
    A_type = hparams.problem.measurement_type

    if A_type not in ['superres', 'inpaint']:
        print("\nCan't save given measurement type\n")
        return

    if A_type == 'inpaint':
        images = get_measurements(None, images, hparams, True)
    elif A_type == 'superres':
        images = get_measurements(None, images, hparams)
        images = get_transpose_measurements(None, images, hparams)

    return images
