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
    def __init__(self, hparams, args, c, A, device=None):
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

        self.A = A
        self._init_net()
        self.register_buffer('c', c.detach().clone()) #TODO remove detach() for e.g.MAML


        self.add_noise = True if hparams.inner.alg == 'langevin' else False

        self.renormalize = hparams.inner.renormalize #whether to re-scale the log likelihood gradient
        if self.renormalize:
            self.rescale_factor = hparams.inner.rescale_factor

        self.verbose = self.hparams.inner.verbose if self.hparams.verbose else False

        if hparams.outer.meta_type != 'mle':
            raise NotImplementedError("Meta Learner type not supported by SGLD!")

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
            raise NotImplementedError


    def forward(self, x_mod, y):
        # for some reason y is a two-channel float. convert to complex
        y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])

        singulars = self.c.clone()
        if self.hparams.problem.measurement_selection:
            singulars = singulars.view(-1, y.shape[-2], y.shape[-1])
            singulars = singulars.repeat(y.shape[-3], 1, 1)
        # print(singulars.shape, x_mod.shape, y.shape)
        self.H_funcs._singulars = torch.sqrt(torch.abs(singulars)).reshape(-1)

        # TODO: make sigma_0 non-zero
        # patch = gt_image[0,:,:5,:5]
        # mag_patch = torch.sum(torch.square(patch), axis=0)
        # sigma_0 = torch.sqrt(torch.mean(mag_patch)).item()
        # print(f'estimated noise is: {sigma_0}')
        # args.sigma_0 = sigma_0
        sigma_0 = 0

        # print('lah', y.shape)
        # print(y)
        x = efficient_generalized_steps(x_mod, self.seq, self.model, self.betas,\
        self.H_funcs, y, sigma_0, etaB=self.args.etaB, etaA=self.args.eta, \
                                        etaC=self.args.eta)
        return x[0][-1].to(self.device)
    # def forward(self, x_mod, y):
    #     fmtstr = "%10i %10.3g %10.3g %10.3g %10.3g %10.3g"
    #     titlestr = "%10s %10s %10s %10s %10s %10s"
    #     if self.verbose:
    #         print('\n')
    #         print(titlestr % ("Noise_Level", "Step_LR", "Meas_Loss", "Score_Norm", "Meas_Grad_Norm", "Total_Grad_Norm"))
    #
    #     step_num = 0
    #
    #     #iterate over noise level index
    #     for t in self.used_levels:
    #         sigma = self.sigmas[t]
    #
    #         labels = torch.ones(x_mod.shape[0]).type_as(x_mod) * t
    #         labels = labels.long()
    #
    #         step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2
    #
    #         for s in range(self.T):
    #             prior_grad = self.model(x_mod, labels)
    #
    #             likelihood_grad = get_likelihood_grad(self.c, y, self.A, x_mod, self.hparams.use_autograd,\
    #                                 1/(sigma**2), self.hparams.outer.exp_params, learn_samples=self.hparams.problem.learn_samples,
    #                                 sample_pattern=self.hparams.problem.sample_pattern)
    #
    #             #TODO rescale this to go sample-by-sample instead of just overall!
    #             if self.renormalize:
    #                 likelihood_grad /= (torch.norm( likelihood_grad ) + 1e-6) #small epsilon to prevent div by 0
    #                 likelihood_grad *= torch.norm( prior_grad )
    #                 likelihood_grad *= self.rescale_factor
    #
    #             grad = prior_grad - likelihood_grad
    #
    #             if self.add_noise:
    #                 noise = torch.randn_like(x_mod)
    #                 x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)
    #             else:
    #                 x_mod = x_mod + step_size * grad
    #
    #             if self.verbose and (step_num % self.verbose == 0 or step_num == self.total_steps - 1):
    #                 with torch.no_grad():
    #                     prior_grad_norm = torch.norm(prior_grad.view(prior_grad.shape[0], -1), dim=-1).mean().item()
    #                     likelihood_grad_norm = torch.norm(likelihood_grad.view(likelihood_grad.shape[0], -1), dim=-1).mean().item()
    #                     grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean().item()
    #                     likelihood_loss = log_cond_likelihood_loss(self.c, y, self.A, x_mod, 2., self.hparams.outer.exp_params,
    #                                         tuple(np.arange(y.dim())[1:]), self.hparams.problem.learn_samples, self.hparams.problem.sample_pattern).mean().item()
    #
    #                     print(fmtstr % (t, step_size, likelihood_loss, prior_grad_norm, likelihood_grad_norm, grad_norm))
    #
    #             step_num += 1
    #
    #     if self.verbose:
    #         print('\n')
    #
    #     x_mod = torch.clamp(x_mod, 0.0, 1.0)
    #
    #     return x_mod
    #

    def set_c(self, c):
        #self.c = c #changing this from parameter to buffer
        self.c = c.detach().clone().type_as(self.c).to(self.c.device) #TODO remove detach() for e.g.MAML

    def _init_net(self):
        """Initializes score net and related attributes"""
        # if self.hparams.net.model != "ncsnv2":
        #     raise NotImplementedError("This model is unsupported!")
        #
        # ckpt_path = self.hparams.net.checkpoint_dir
        # config_path = self.hparams.net.config_file
        #
        # with open(config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        # net_config = dict2namespace(config)
        # net_config.device = self.hparams.device
        #
        # states = torch.load(ckpt_path, map_location=self.hparams.device)
        #
        # if self.hparams.data.dataset == 'ffhq':
        #     test_score = NCSNv2Deepest(net_config).to(self.hparams.device)
        # elif self.hparams.data.dataset == 'celeba':
        #     test_score = NCSNv2(net_config).to(self.hparams.device)
        #
        # test_score = torch.nn.DataParallel(test_score)
        # test_score.load_state_dict(states[0], strict=True)
        #
        # if net_config.model.ema:
        #     ema_helper = EMAHelper(mu=net_config.model.ema_rate)
        #     ema_helper.register(test_score)
        #     ema_helper.load_state_dict(states[-1])
        #     ema_helper.ema(test_score)
        #
        # model = test_score.module.to(self.hparams.device)
        # sigmas = get_sigmas(net_config).to(self.hparams.device)
        #
        # model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # self.model = model
        # self.register_buffer('sigmas', sigmas)

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
        self.model = model

    def _get_decimated_sigmas(self):
        decimate = self.hparams.inner.decimation_factor
        decimate_type = self.hparams.inner.decimation_type

        L = len(self.sigmas)
        num_used_levels = L // decimate

        #geometrically-spaced entries biased towards later noise levels
        if decimate_type == 'log_last':
            used_levels = np.ceil(-np.geomspace(start=L, stop=1, num=num_used_levels)+L).astype(np.long)
        #geometrically-spaced entries biased towards earlier noise levels
        elif decimate_type == 'log_first':
            used_levels = (np.geomspace(start=1, stop=L, num=num_used_levels)-1).astype(np.long)
        #grab just the last few noise levels
        elif decimate_type == 'last':
            used_levels = np.arange(L)[-num_used_levels:].astype(np.long)
        #grab just the first few noise levels
        elif decimate_type == 'first':
            used_levels = np.arange(L)[:num_used_levels].astype(np.long)
        #grab equally-spaced levels
        elif decimate_type == 'linear':
            used_levels = np.ceil(np.linspace(start=0, stop=L-1, num=num_used_levels)).astype(np.long)
        else:
            raise NotImplementedError("Decimation type not supported!")

        return used_levels

# x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
#             etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta)
