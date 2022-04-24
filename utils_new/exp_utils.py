import torch
import random
import numpy as np
import argparse
import yaml
from datetime import datetime
from PIL import Image
import os
import pickle

def save_image(image, path):
    """Save a pytorch image as a png file"""
    image = image.detach().cpu().numpy() #image comes in as an [C, H, W] torch tensor
    x_png = np.uint8(np.clip(image*256,0,255))
    x_png = x_png.transpose(1,2,0)
    if x_png.shape[-1] == 1:
        x_png = x_png[:,:,0]
    x_png = Image.fromarray(x_png).save(path)

def save_images(images, labels, save_prefix):
    """Save a batch of images (in a dictionary) to png files"""
    for image_num, image in zip(labels, images):
        if isinstance(image_num, torch.Tensor):
            save_image(image, os.path.join(save_prefix, str(image_num.item())+'.png'))
        elif isinstance(image_num, int):
            save_image(image, os.path.join(save_prefix, str(image_num)+'.png'))
        elif isinstance(image_num, str):
            save_image(image, os.path.join(save_prefix, image_num+'.png'))
        else:
            raise NotImplementedError("Bad type given to save_images for labels.")

def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data

def set_all_seeds(random_seed: int):
    """
    Sets random seeds in numpy, torch, and random.

    Args:
        random_seed: The seed to set.
                     Type: int.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    return

def dict2namespace(config: dict):
    """
    Converts a given dictionary to an argparse namespace object.

    Args:
        config: The dictionary to convert to namespace.
                Type: dict.

    Returns:
        namespace: The converted namespace.
                   Type: Namespace.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_config(config_path):
    """Parses experiment configuration yaml file.
       Does not perform argument validation for types or values."""
    with open(config_path, 'r') as f:
        hparams = yaml.safe_load(f)

    #set up the devices - account for multi_GPU DP mode
    if hparams['use_gpu']:
        num = hparams['gpu_num']
        if num == -1:
            hparams['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if torch.cuda.device_count() > 1 and hparams['verbose']:
                print("Let's use ", torch.cuda.device_count(), " GPUs!")
        else:
            hparams['device'] = torch.device('cuda:'+str(num)) if torch.cuda.is_available() else torch.device('cpu:0')
    else:
        hparams['gpu_num'] = None
        hparams['device'] = torch.device('cpu:0')

    #account for debugging
    if hparams['debug']:
        hparams['save_imgs'] = False
        hparams['save_dir'] = None

    #account for resumption
    if not hparams['resume']:
        hparams['resume_dir'] = None

    #NET
    #check generative models
    if hparams['net']['model'] != 'ncsnv2':
        raise NotImplementedError("No implementation for this network yet!")

    #DATA
    #check dataset
    if hparams['data']['dataset'] not in ["celeba", "ffhq"]:
        raise NotImplementedError("This dataset has not been implemented!")

    #make some more input dimension metadata
    hparams['data']['image_shape'] = (hparams['data']['num_channels'], hparams['data']['image_size'], hparams['data']['image_size'])
    hparams['data']['n_input'] = int(np.prod(hparams['data']['image_shape']))

    #OUTER
    #check meta learning type
    if hparams['outer']['meta_type'] != 'mle':
        raise NotImplementedError("This Meta Learning Algorithm has not been implemented yet!")
    
    if hparams['outer']['meta_loss_type'] != 'l2':
        raise NotImplementedError("This Meta Loss has not been implemented yet!")

    #set reg hyperparam stuff
    if not hparams['outer']['reg_hyperparam']:
        hparams['outer']['reg_hyperparam_type'] = None
        hparams['outer']['reg_hyperparam_scale'] = None
    else:
        if hparams['outer']['reg_hyperparam_type'] not in ['l1', 'soft', 'hard']:
            raise NotImplementedError("This Meta Regularization term has not been implemented yet!")

    #automatically set ROI to a region if not specified
    #TODO implement ROI!
    if hparams['outer']['ROI_loss'] and not isinstance(hparams['outer']['ROI'], tuple):
        if hparams['data']['dataset'] == "celeba":
            hparams['outer']['ROI'] = ((27, 15),(35, 35))
        elif hparams['data']['dataset'] == "ffhq":
            hparams['outer']['ROI'] = ((90, 50),(60, 156))
        else:
            raise NotImplementedError("You must provide an ROI for this dataset")

    #OPT
    #check if opt supported
    if hparams['opt']['optimizer'] not in ['adam', 'sgd']:
        raise NotImplementedError("This optimizer has not been implemented yet!")
    
    #null decay if needed
    if not hparams['opt']['decay']:
        hparams['opt']['lr_decay'] = None
        hparams['opt']['decay_on_val'] = False

    #INNER
    #check if algorithm supported
    if hparams['inner']['alg'] not in ['langevin', 'map']:
        raise NotImplementedError("This reconstruction algorithm has not been implemented yet!")
    
    #null decimation stuff if needed
    if not hparams['inner']['decimate']:
        hparams['inner']['decimation_factor'] = None
        hparams['inner']['decimation_type'] = None

    #PROBLEM
    #check if problem supported
    if hparams['problem']['measurement_type'] not in ['superres', 'inpaint', 'identity', 'gaussian', 'fourier']:
        raise NotImplementedError("This forward operator has not been implemented")

    #do problem-specific stuff
    #num measurements will refer to the number of total measurements (i.e. including each channel as a separate meas)
    #y_shape will refer to the shape of the measurements during training time
    if hparams['problem']['measurement_type'] == 'superres':
        new_size = hparams['data']['image_size'] // hparams['problem']['downsample_factor']
        hparams['problem']['num_measurements'] = hparams['data']['num_channels'] * (new_size**2)

    elif hparams['problem']['measurement_type'] == 'identity':
        hparams['problem']['num_measurements'] = hparams['data']['n_input']

    elif hparams['problem']['measurement_type'] == 'inpaint':
        if not hparams['problem']['inpaint_random']:
            hparams['problem']['num_measurements'] = hparams['data']['n_input'] - hparams['data']['num_channels'] * hparams['problem']['inpaint_size']**2
        else:
            hparams['problem']['num_measurements'] = hparams['problem']['num_measurements'] * hparams['data']['num_channels'] #specified in number of pixels, but here we multiply by 3 since all color channels are kept

    elif hparams['problem']['measurement_type'] == 'fourier':
        if hparams['problem']['fourier_mask_type'] != 'random':
            raise NotImplementedError("This Fourier mask type is not implemented yet!")
        else:
            hparams['problem']['num_measurements'] = hparams['problem']['num_measurements'] * hparams['data']['num_channels'] #specified in number of pixels, but here we multiply by 3 since all color channels are kept

    #make y shape (Gaussian is well-behaved and needs no specifics)
    if hparams['problem']['measurement_type'] != 'fourier':
        hparams['problem']['y_shape'] = (hparams['problem']['num_measurements'])
    else:
        hparams['problem']['y_shape'] = (hparams['problem']['num_measurements'] * 2)
    
    #deal with sampling and coupling of pixels
    if hparams['problem']['learn_samples']:
        if hparams['problem']['measurement_type'] not in ['superres', 'inpaint', 'fourier']:
            raise NotImplementedError("Sample selection not supported for chosen measurement type!")
        elif hparams['outer']['hyperparam_type'] != "vector":
            raise NotImplementedError("Sample selection must use a vector hyperparam")

        if hparams['problem']['measurement_type'] in ['inpaint', 'fourier']:
            hparams['problem']['num_measurements'] = hparams['data']['n_input']
        
        if hparams['problem']['measurement_type'] == 'superres':
            hparams['problem']['y_shape'] = (hparams['data']['num_channels'], new_size, new_size)
        elif hparams['problem']['measurement_type'] == 'inpaint':
            hparams['problem']['y_shape'] = hparams['data']['image_shape']
        elif hparams['problem']['measurement_type'] == 'fourier':
            hparams['problem']['y_shape'] = hparams['data']['image_shape'] + (2,) #2 at the end for real/im
        
        if hparams['problem']['sample_pattern'] not in ['horizontal', 'vertical', 'random']:
            raise NotImplementedError("Given sample pattern not implemented!")

    #deal with noise params
    if not hparams['problem']['add_noise']:
        hparams['problem']['noise_type'] = None
        hparams['problem']['noise_std'] = None

    if hparams['problem']['add_dependent_noise']:
        raise NotImplementedError("Dependent noise not supported yet!")
    hparams['problem']['add_dependent_noise'] = False
    hparams['problem']['dependent_noise_type'] = None
    hparams['problem']['dependent_noise_std'] = None

    #create namespace, print, and wrap up
    HPARAMS = dict2namespace(hparams)
    print(yaml.dump(HPARAMS, default_flow_style=False))

    return HPARAMS

def parse_args(docstring="", manual=False, config=None, doc=None):
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
