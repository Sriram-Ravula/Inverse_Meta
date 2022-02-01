from pickle import FALSE
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

from utils.alg_utils import get_decimated_sigmas
from utils import loss_utils

class SGLD_NSCNv2(nn.Module):
    def __init__(self, c, A, model, sigmas, hparams, efficient_inp=False):
        super().__init__()
        self.c = torch.nn.Parameter(c, requires_grad=c.requires_grad)
        self.A = torch.nn.Parameter(A, requires_grad=A.requires_grad)
        self.model = model
        self.sigmas = torch.nn.Parameter(sigmas, requires_grad=sigmas.requires_grad)
        self.hparams = hparams
        self.efficient_inp = efficient_inp

        self.T = hparams.inner.T
        self.step_lr = hparams.inner.lr
        self.decimate = hparams.inner.decimation_factor if hparams.inner.decimation_factor > 0 else False
        self.add_noise = True if hparams.inner.alg == 'langevin' else False
        self.maml_use_last = hparams.outer.maml_use_last
        self.verbose = hparams.outer.verbose

        if self.verbose:
            self.verbose = hparams.inner.verbose if hparams.inner.verbose > 0 else False

        if self.decimate:
            self.used_levels = get_decimated_sigmas(len(sigmas), hparams)
            self.total_steps = len(self.used_levels) * self.T
        else:
            self.total_steps = len(sigmas) * self.T
            self.used_levels = np.arange(len(sigmas))

        if self.maml_use_last not in np.arange(start=1, stop=self.total_steps) or hparams.outer.meta_type != 'maml':
            self.maml_use_last = False

        if hparams.outer.meta_type == 'maml' and not self.maml_use_last:
            self.create_graph = True  
        else:
            self.create_graph = False

    def forward(self, x_mod, y, eval):
        grad_flag_x = x_mod.requires_grad
        grad_flag_c = self.c.requires_grad

        if eval or not self.create_graph:
            x_mod.requires_grad_(False) 
            self.c.requires_grad_(False)    

        fmtstr = "%10i %10.3g %10.3g %10.3g %10.3g %10.3g"
        titlestr = "%10s %10s %10s %10s %10s %10s"
        if self.verbose:
            print('\n') 
            print(titlestr % ("Noise_Level", "Step_LR", "Meas_Loss", "Score_Norm", "Meas_Grad_Norm", "Total_Grad_Norm"))

        step_num = 0

        #iterate over noise level index
        for t in self.used_levels:
            sigma = self.sigmas[t]

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * t
            labels = labels.long()

            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2

            for s in range(self.T):
                if not eval and (not self.create_graph and self.hparams.outer.meta_type == 'maml'): 
                    if (self.total_steps - step_num) == self.maml_use_last:
                        self.create_graph = True  
                        x_mod.requires_grad_()
                        self.c.requires_grad_() 
                        if self.verbose:
                            print("\nStarting to track MAML gradient at iter " + str(step_num) + '\n')

                prior_grad = self.model(x_mod, labels)

                likelihood_grad = loss_utils.get_likelihood_grad(self.c, y, self.A, x_mod, self.hparams, 1/(sigma**2), self.efficient_inp,\
                    retain_graph = self.create_graph if not eval else False, \
                    create_graph = self.create_graph if not eval else False)

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
                        likelihood_loss = loss_utils.simple_likelihood_loss(y, self.A, x_mod, self.hparams, self.efficient_inp).mean()

                        print(fmtstr % (t, step_size, likelihood_loss, prior_grad_norm, likelihood_grad_norm, grad_norm))
        
                step_num += 1
        
        if self.verbose:
            print('\n')
        
        x_mod = torch.clamp(x_mod, 0.0, 1.0)
    
        x_mod.requires_grad_(grad_flag_x)
        self.c.requires_grad_(grad_flag_c)

        return x_mod
