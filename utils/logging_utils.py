"""
Class used to help with tensorboard logging, checkpointing and loading, and statistics saving.
"""
import torch.utils.tensorboard as tb
import torch
import numpy as np
import sys
import os
import yaml
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
import pickle


from utils.metrics_utils import Metrics
from meta_learner import MetaLearner
from loss_utils import get_inpaint_mask

def save_image(image, path):
    """Save a pytorch image as a png file"""
    image = image.detach().cpu().numpy() #image comes in as an [C, H, W] torch tensor
    x_png = np.uint8(np.clip(image*256,0,255))
    x_png = x_png.transpose(1,2,0)
    if x_png.shape[-1] == 1:
        x_png = x_png[:,:,0]
    x_png = Image.fromarray(x_png).save(path)

def save_images(est_images, save_prefix):
    """Save a batch of images (in a dictionary) to png files"""
    for image_num, image in est_images.items():
        save_image(image, os.path.join(save_prefix, str(image_num), '.png'))

def save_measurement_images(est_images, hparams, save_prefix):
    """Save a batch of image measurements to png files"""
    A_type = hparams.problem.measurement_type

    if A_type not in ['superres', 'inpaint']:
        print("Can't save given measurement type")
        return

    for image_num, image in est_images.items():
        if A_type == 'superres':
            image = image * get_inpaint_mask(hparams)
        elif A_type == 'inpaint':
            image = F.avg_pool2d(image, hparams.problem.downsample_factor)
            image = F.interpolate(image, scale_factor=hparams.problem.downsample_factor)
        save_image(image, os.path.join(save_prefix, str(image_num), '.png'))

def get_img_matrix(images):
    return torchvision.utils.make_grid(images.detach().cpu().numpy(), nrow=8)

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

class Logger:
    """
    Class for saving, checkpointing, and logging various things
    """
    def __init__(self, metrics: Metrics, learner: MetaLearner, hparams, log_dir):
        self.log_dir = log_dir
        self.hparams = hparams
        self.image_root = os.path.join(log_dir, 'images')
        self.tb_root = os.path.join(log_dir, 'tensorboard')
        self.metrics_root = os.path.join(log_dir, 'metrics')

        self.metrics = metrics
        self.learner = learner

        self.__make_log_folder()
        self.__save_config()

        self.tb_logger = tb.SummaryWriter(log_dir=log_dir)

    def __make_log_folder(self):
        if os.path.exists(self.log_dir):
            print("Folder exists. Program halted.")
            sys.exit(0)
        else:
            os.makedirs(self.log_dir)
            os.makedirs(self.image_root)
            os.makedirs(self.tb_root)
            os.makedirs(self.metrics_root)

    def __save_config(self):
        with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False)
    
    def checkpoint(self):
        save_to_pickle(self.metrics, os.path.join(self.metrics_root, str(self.learner.global_iter)))
        #TODO save everything from learner but the net, dataset, and optimizers in pickle

        return
    
    def save_image_measurments(self, images, image_nums, save_prefix):
        save_path = os.path.join(self.image_root, save_prefix)

        image_dict = {}

        for i in range(images.shape[0]):
            image_dict[image_nums[i]] = images[i]
        
        save_measurement_images(image_dict, self.hparams, save_path)

    def save_images(self, images, image_nums, save_prefix):
        save_path = os.path.join(self.image_root, save_prefix)

        image_dict = {}

        for i in range(images.shape[0]):
            image_dict[image_nums[i]] = images[i]
        
        save_images(image_dict, save_path)





