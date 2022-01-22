import torch
import torch.nn.functional as F
import numpy as np


def get_ssim(x_hat, x, hparams):
    """
    Calculates SSIM(x_hat, x) and MS-SSIM(x_hat, x)
    If batch dimension is larger than 1, return the mean value
    """

def get_nmse(x_hat, x, hparams):
    """
    Calculate ||x_hat - x|| / ||x||
    If batch dimension is larger than 1, return the mean value
    """

def get_psnr(x_hat, x, hparams):
    """
    Calculate 
    """