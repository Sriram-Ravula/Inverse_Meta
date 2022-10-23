import torchvision
import numpy as np
import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.fft as torch_fft
from torch import nn
import os
import random
import yaml

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest

from datasets.mri_dataloaders import get_mvue
from problems.fourier_multicoil import MulticoilForwardMRINoMask
from utils_new.exp_utils import dict2namespace

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling

def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling

class Dummy:
    def __init__(self):
        self.s_maps = None

class NCSNv2(torch.nn.Module):
    def __init__(self, hparams, args, c, device=None):
        super().__init__()
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.hparams = hparams
        self.args = args

        with open(self.hparams.net.config_dir, 'r') as f:
            ncsn_config = yaml.safe_load(f)
        self.ncsn_config = dict2namespace(ncsn_config)
        self.langevin_config = self.ncsn_config.langevin_config 
        self.langevin_config.device = self.device

        self._init_net()
        self.register_buffer('c', c.detach().clone()) #TODO remove detach() for e.g.MAML

        self.H_funcs = Dummy()
    
    def forward(self, x_mod, y, sigma_0=0):
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])
        
        mask = self.c.clone()
        maps = self.H_funcs.s_maps.clone()
        samples = x_mod.clone()

        ref = mask[None, None, :, :] * y

        estimated_mvue = torch.tensor(
                    get_mvue(ref.cpu().numpy(),
                    maps.cpu().numpy()), device=ref.device)

        pbar = tqdm(range(self.langevin_config.model.num_classes))
        pbar_labels = ['class', 'step_size', 'error', 'mean', 'max']

        step_lr = self.langevin_config.sampling.step_lr
        forward_operator = MulticoilForwardMRINoMask(maps)

        with torch.no_grad():
            for c in pbar:
                if c <= self.hparams.net.start_iter:
                    continue
                elif c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.langevin_config.sampling.n_steps_each
                
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    noise = torch.randn_like(samples) * np.sqrt(step_size * 2)

                    p_grad = self.score(samples, labels)

                    meas = forward_operator(normalize(samples, estimated_mvue))
                    meas = mask[None, None, :, :] * meas

                    meas_grad = torch.view_as_real(torch.sum(self._ifft(meas-ref) * torch.conj(maps), axis=1) ).permute(0,3,1,2)

                    meas_grad = unnormalize(meas_grad, estimated_mvue)
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor)
                    meas_grad /= torch.norm( meas_grad )
                    meas_grad *= torch.norm( p_grad )
                    meas_grad *= self.hparams.net.mse

                    samples = samples + step_size * (p_grad - meas_grad) + noise

                    # compute metrics
                    metrics = [c, step_size, (meas-ref).norm(), (p_grad-meas_grad).abs().mean(), (p_grad-meas_grad).abs().max()]
                    self.update_pbar_desc(pbar, metrics, pbar_labels)
                    # if nan, break
                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)

        return normalize(samples, estimated_mvue)

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x    

    def update_pbar_desc(self, pbar, metrics, labels):
        pbar_string = ''
        for metric, label in zip(metrics, labels):
            pbar_string += f'{label}: {metric:.7f}; '
        pbar.set_description(pbar_string)

    def set_c(self, c):
        self.c = c.detach().clone().type_as(self.c).to(self.c.device) #TODO remove detach() for e.g.MAML
    
    def _init_net(self):
        self.score = NCSNv2Deepest(self.langevin_config).to(self.device)

        self.sigmas_torch = get_sigmas(self.langevin_config)
        self.sigmas = self.sigmas_torch.cpu().numpy()

        states = torch.load(self.hparams.net.checkpoint_dir)

        self.score = torch.nn.DataParallel(self.score)

        self.score.load_state_dict(states[0], strict=True)
        if self.langevin_config.model.ema:
            ema_helper = EMAHelper(mu=self.langevin_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states

        for param in self.score.parameters():
            param.requires_grad = False
        self.score.eval()

        self.score = self.score.module.to(self.device)
        