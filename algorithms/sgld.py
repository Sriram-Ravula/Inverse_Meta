import numpy as np
import torch
from time import time
import yaml

from ncsnv2.models.ncsnv2 import NCSNv2, NCSNv2Deepest
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models import get_sigmas

from utils_new.exp_utils import dict2namespace
from utils_new.inner_loss_utils import log_cond_likelihood_loss, get_likelihood_grad
from problems import get_forward_operator

class SGLD_NCSNv2(torch.nn.Module):
    def __init__(self, hparams, c, A):
        super().__init__()
        self.hparams = hparams

        #These are module so they move devices automatically
        self.A = A
        self.model, sigmas = self._init_net()
        self.c = c
        
        #other params
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('T', self.hparams.inner.T)
        self.register_buffer('step_lr', self.hparams.inner.lr)
        if self.hparams.inner.decimate:
            self.register_buffer('used_levels', self._get_decimated_sigmas())
            self.register_buffer('total_steps', len(self.used_levels) * self.T)
        else:
            self.register_buffer('total_steps', len(self.sigmas) * self.T)
            self.register_buffer('used_levels', np.arange(len(self.sigmas)))
        self.add_noise = True if hparams.inner.alg == 'langevin' else False

        self.verbose = self.hparams.inner.verbose if self.hparams.verbose else False

        if hparams.outer.meta_type != 'mle':
            raise NotImplementedError("Meta Learner type not supported by SGLD!")
    
    def forward(self, x_mod, y, eval=False):
        grad_flag_x = x_mod.requires_grad
        grad_flag_c = self.c.requires_grad

        if eval:
            x_mod = x_mod.detach() 
            self.c = self.c.detach() 
        
        fmtstr = "%10i %10.3g %10.3g %10.3g %10.3g %10.3g"
        titlestr = "%10s %10s %10s %10s %10s %10s"
        if self.verbose:
            print('\n') 
            print(titlestr % ("Noise_Level", "Step_LR", "Meas_Loss", "Score_Norm", "Meas_Grad_Norm", "Total_Grad_Norm"))

        step_num = 0

        #iterate over noise level index
        for t in self.used_levels:
            sigma = self.sigmas[t]

            labels = torch.ones(x_mod.shape[0]).type_as(x_mod) * t
            labels = labels.long()

            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2

            for s in range(self.T):
                prior_grad = self.model(x_mod, labels)

                likelihood_grad = get_likelihood_grad(self.c, y, self.A, x_mod, self.hparams.use_autograd,\
                                    1/(sigma**2), self.hparams.outer.exp_params)

                grad = prior_grad - likelihood_grad

                if self.add_noise:
                    noise = torch.randn_like(x_mod)
                    x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)
                else:
                    x_mod = x_mod + step_size * grad

                if self.verbose and (step_num % self.verbose == 0 or step_num == self.total_steps - 1):
                    with torch.no_grad():
                        prior_grad_norm = torch.norm(prior_grad.view(prior_grad.shape[0], -1), dim=-1).mean().item()
                        likelihood_grad_norm = torch.norm(likelihood_grad.view(likelihood_grad.shape[0], -1), dim=-1).mean().item()
                        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean().item()
                        likelihood_loss = log_cond_likelihood_loss(1, y, self.A, x_mod, reduce_dims=(1)).mean()

                        print(fmtstr % (t, step_size, likelihood_loss, prior_grad_norm, likelihood_grad_norm, grad_norm))
        
                step_num += 1
        
        if self.verbose:
            print('\n')
        
        x_mod = torch.clamp(x_mod, 0.0, 1.0)
    
        x_mod.requires_grad_(grad_flag_x)
        self.c.requires_grad_(grad_flag_c)

        return x_mod
        
    def set_c(self, c):
        self.register_buffer('c', c)

    def _init_net(self):
        """Initializes score net and related attributes"""
        if self.hparams.net.model != "ncsnv2":
            raise NotImplementedError("This model is unsupported!") 
        
        if self.hparams.verbose:
            print("\nINITIALIZING NETWORK\n")
            start = time()

        ckpt_path = self.hparams.net.checkpoint_dir
        config_path = self.hparams.net.config_file

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        net_config = dict2namespace(config)
        net_config.device = self.hparams.device

        states = torch.load(ckpt_path, map_location=self.hparams.device)

        if self.hparams.data.dataset == 'ffhq':
            test_score = NCSNv2Deepest(net_config).to(self.hparams.device)
        elif self.hparams.data.dataset == 'celeba':
            test_score = NCSNv2(net_config).to(self.hparams.device)
        
        test_score = torch.nn.DataParallel(test_score)
        test_score.load_state_dict(states[0], strict=True)
        test_score.to(self.hparams.device) #second .to() to make sure we are utilising all GPUs

        if net_config.model.ema:
            ema_helper = EMAHelper(mu=net_config.model.ema_rate)
            ema_helper.register(test_score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(test_score)

        test_score.eval()
        for param in test_score.parameters():
            param.requires_grad = False

        model = test_score.module.cpu()
        sigmas = get_sigmas(net_config).cpu()

        if self.hparams.verbose:
            end = time()
            print("\nNET TIME: ", str(end - start), "S\n")

        return model, sigmas

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
            used_levels = np.arange(L)[-num_used_levels:]
        #grab just the first few noise levels
        elif decimate_type == 'first':
            used_levels = np.arange(L)[:num_used_levels]
        #grab equally-spaced levels
        elif decimate_type == 'linear':
            used_levels = np.ceil(np.linspace(start=0, stop=L-1, num=num_used_levels)).astype(np.long)
        else:
            raise NotImplementedError("Decimation type not supported!")

        return used_levels