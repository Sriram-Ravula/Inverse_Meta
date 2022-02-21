import torch
import random
import numpy as np
import argparse
import yaml
from datetime import datetime

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
    hparams['data']['n_input'] = np.prod(hparams['data']['image_shape'])

    #OUTER
    #check meta learning type
    if hparams['outer']['meta_type'] not in ['implicit', 'maml', 'mle']:
        raise NotImplementedError("This Meta Learning Algorithm has not been implemented yet!")
    
    if hparams['outer']['meta_loss_type'] != 'l2' or hparams['outer']['measurement_loss']:
        raise NotImplementedError("This Meta Loss has not been implemented yet!")

    #automatically set ROI to a region if not specified
    if hparams['outer']['ROI_loss'] and not isinstance(hparams['outer']['ROI'], tuple):
        if hparams['data']['dataset'] == "celeba":
            hparams['outer']['ROI'] = ((27, 15),(35, 35))
        elif hparams['data']['dataset'] == "ffhq":
            hparams['outer']['ROI'] = ((90, 50),(60, 156))
    
    #set reg hyperparam stuff
    if not hparams['outer']['reg_hyperparam']:
        hparams['outer']['reg_hyperparam_type'] = None
        hparams['outer']['reg_hyperparam_scale'] = None
    else:
        if hparams['outer']['reg_hyperparam_type'] != 'l1':
            raise NotImplementedError("This Meta Regularization term has not been implemented yet!")
    
    #set finite difference stuff
    if not hparams['outer']['finite_difference']:
        hparams['outer']['finite_difference_coeff'] = None

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
    if hparams['problem']['measurement_type'] == 'superres':
        hparams['problem']['num_measurements'] = hparams['data']['num_channels'] * (hparams['data']['image_size']//hparams['problem']['downsample_factor'])**2

    elif hparams['problem']['measurement_type'] == 'inpaint':
        if hparams['problem']['inpaint_random']:
            raise NotImplementedError("Random inpainting not apploed yet!")
            #hparams['problem']['inpaint_size'] = None
        else:
            hparams['problem']['num_measurements'] = hparams['data']['n_input'] - hparams['data']['num_channels'] * hparams['problem']['inpaint_size']**2

    elif hparams['problem']['measurement_type'] == 'identity':
        hparams['problem']['num_measurements'] = hparams['data']['n_input']

    elif hparams['problem']['measurement_type'] == 'fourier':
        if hparams['problem']['fourier_mask_type'] != 'random':
            raise NotImplementedError("This Fourier mask type is not implemented yet!")

    hparams['problem']['y_shape'] = (hparams['problem']['num_measurements'])

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
