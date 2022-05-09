import numpy as np
import os
import torch
from time import time
import yaml
import copy

from ddrm_mri.models.ncsnpp import NCSNpp
from ddrm_mri.models.ema import ExponentialMovingAverage
from ddrm_mri.runners.diffusion import get_beta_schedule
from ddrm_mri.functions.denoising import efficient_generalized_steps
from ddrm_mri.functions.ckpt_util import get_ckpt_path
from ddrm_mri.functions.svd_replacement import FourierSingleCoil, FourierMultiCoil
from ddrm_mri.ncsnpp_utils import restore_checkpoint

from utils_new.exp_utils import dict2namespace
# from utils_new.inner_loss_utils import log_cond_likelihood_loss, get_likelihood_grad

class DDRM(torch.nn.Module):
    def __init__(self, hparams, args, c, device=None):
        """
        Making this a module so we can parallelize operations for the
            - score network (prior)
            - forward operator (likelihood)
            - meta parameter (c)
        If each component listed were to be parallelized separately (i.e. as part of different forward() calls),
            PyTorch would use more memory transfers between CPU and GPU.
        """
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

        self._init_net()
        self.register_buffer('c', c.detach().clone()) #TODO remove detach() for e.g.MAML

        # special stuff for the Variance Preserving NCSN++
        self.model_var_type = self.hparams.model.var_type
        betas = get_beta_schedule(hparams.diffusion.beta_schedule,
                                       hparams.diffusion.beta_start,
                                       hparams.diffusion.beta_end,
                                       hparams.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)

        start = 0
        end = self.betas.shape[0]
        skip = (end - start)// self.args.timesteps
        self.seq = np.array(range(start, end, skip))
        self.register_buffer('used_levels', torch.from_numpy(self.seq))

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # stuff for measurement operator and its SVD
        if self.hparams.problem.measurement_type == 'fourier-multicoil':
            R = self.hparams.problem.R
            pattern = self.hparams.problem.pattern
            orientation = self.hparams.problem.orientation
            self.H_funcs = FourierMultiCoil(self.hparams.data.image_size, R, pattern,
                                        orientation, self.device)
        else:
            raise NotImplementedError("This forward operator is not supported")


    def forward(self, x_mod, y, sigma_0=0):
        # for some reason y is a two-channel float. convert to complex
        print("DDRM Y Shape: ", y.shape)
        print("DDRM Y Type: ", y.dtype)
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])

        singulars = self.c.clone()
        # singulars = torch.ones_like(self.c)
        singulars = singulars.view(-1, y.shape[-2], y.shape[-1])
        singulars = singulars.repeat(y.shape[-3], 1, 1)
        c = self.c.clone().view(y.shape[2:])
        # c = torch.ones_like(self.c).view(y.shape[2:])
        print(c[None, None, :, :].shape)
        y_ = torch.sqrt(torch.abs(c[None, None, :, :])) * y
        # print(singulars.shape, x_mod.shape, y.shape)
        self.H_funcs._singulars = torch.sqrt(torch.abs(singulars)).reshape(-1)
        print(self.H_funcs._singulars)

        # TODO: make sigma_0 non-zero
        # patch = gt_image[0,:,:5,:5]
        # mag_patch = torch.sum(torch.square(patch), axis=0)
        # sigma_0 = torch.sqrt(torch.mean(mag_patch)).item()
        # print(f'estimated noise is: {sigma_0}')
        # args.sigma_0 = sigma_0
        # sigma_0 = 0

        # print('lah', y.shape)
        # print(y)
        x = efficient_generalized_steps(x_mod, self.seq, self.model, self.betas,\
        self.H_funcs, y_, sigma_0, etaB=self.args.etaB, etaA=self.args.eta, \
                                        etaC=self.args.eta)
        return x[0][-1].to(self.device)

    def set_c(self, c):
        #self.c = c #changing this from parameter to buffer
        self.c = c.detach().clone().type_as(self.c).to(self.c.device) #TODO remove detach() for e.g.MAML

    def _init_net(self):
        """Initializes score net and related attributes"""
        if self.hparams.net.model != "ncsnpp":
            raise NotImplementedError("This model is unsupported!")

        model = NCSNpp(self.hparams)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        ema = ExponentialMovingAverage(model.parameters(), decay=self.hparams.model.ema_rate)
        state = dict(model=model, ema=ema)

        ckpt = os.path.join(self.hparams.net.checkpoint_dir, f'checkpoint_{self.hparams.net.checkpoint}.pth')
        state = restore_checkpoint(ckpt, state, self.device)

        if self.hparams.gpu_num == -1:
            self.model = model
        else:
            self.model = model.module
