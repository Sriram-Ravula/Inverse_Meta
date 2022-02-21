import torch
import numpy as np

from algorithms.sgld import SGLD_NCSNv2
from problems import get_forward_operator
from utils.utils import get_measurement_images

from utils_new.meta_utils import hessian_vector_product as hvp
from utils_new.meta_loss_utils import meta_loss, get_meta_grad
from utils_new.inner_loss_utils import get_likelihood_grad

class GBML(torch.nn.Module):
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        self.register_parameter('c', self._init_c())
        
        self.A = get_forward_operator(self.hparams) #module, don't need to register
        self.A = self.A.to(self.device)

        self.langevin_runner = SGLD_NCSNv2(self.hparams, self.c, self.A)
        if self.hparams.gpu_num == -1:
            self.langevin_runner = torch.nn.DataParallel(self.langevin_runner)
            self.langevin_runner = self.langevin_runner.to(self.device)

        self.opt, self.scheduler = self._get_meta_optimizer()
    
    def get_meas_image(self, x, targets=False):
        return self.A.get_measurements_image(x, targets=targets)
    
    def forward(self, x, x_mod=None, eval=False):
        #(1) Find x(c) by running the inner optimization
        x = x.to(self.device)
        y = self.A.forward(x)

        if x_mod is None:
            x_mod = torch.rand(x.shape, device=self.device, requires_grad=True)
        
        x_hat = self.langevin_runner.forward(x_mod, y, eval=eval)

        if self.hparams.outer.meta_type == 'mle':
            meta_grad += self.mle_step(x_hat, x, y)
        else: 
            raise NotImplementedError("Only MLE learning implemented!")

        return x_hat.detach()
    
    def opt_step(self, meta_grad):
        """sets c.grad to False and True"""
        self.meta_opt.zero_grad()

        self.c.requires_grad_()
        if type(self.c.grad) == type(None): #dummy update to make sure grad is initialized
            dummy_loss = torch.sum(self.c)
            dummy_loss.backward()
        self.c.grad.copy_(meta_grad)
        self.opt.step()
        self.c.requires_grad_(False)

        self.c.clamp_(min=0.)
    
    def sched_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def mle_step(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an MLE step.
        grad_c(meta_loss) = - grad_x_c[recon_loss] * (meta_loss)
        (1) Find meta loss
        (2) HVP of grad_x_c(recon_loss) * grad_x(meta_loss)
        """
        #(1)
        grad_x_meta_loss, grad_c_meta_loss = get_meta_grad(x_hat=x_hat,
                                                            x_true=x,
                                                            c = self.c,
                                                            measurement_loss=self.hparams.outer.measurement_loss,
                                                            meta_loss_type=self.hparams.outer.meta_loss_type,
                                                            reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                                            reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                                            reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale,
                                                            ROI_loss=self.hparams.outer.ROI_loss,
                                                            ROI=self.hparams.outer.ROI,
                                                            use_autograd=self.hparams.use_autograd)
        
        #(2)
        self.c.requires_grad_()
        
        cond_log_grad = get_likelihood_grad(self.c, y, self.A, x_hat, self.hparams.use_autograd, 
                                            exp_params=self.hparams.outer.exp_params, retain_graph=True, 
                                            create_graph=True)
        
        out_grad = 0.0
        out_grad -= hvp(self.c, cond_log_grad, grad_x_meta_loss, self.hparams)
        out_grad += grad_c_meta_loss

        return out_grad

    def _init_c(self):
        """
        Initializes the hyperparameters as a scalar, vector, or matrix.

        Args:
            hparams: The experiment parameters to use for init.
                    Type: Namespace.
                    Expected to have the consituents:
                        hparams.outer.hyperparam_type - str in [scalar, vector, matrix]
                        hparams.problem.num_measurements - int
                        hparams.outer.hyperparam_init - int or float
        
        Returns:
            c: The initialized hyperparameters.
            Type: Tensor.
            Shape: [] for scalar hyperparam
                    [m] for vector hyperparam
                    [m,m] for matrix hyperparam
        """
        c_type = self.hparams.outer.hyperparam_type
        m = self.hparams.problem.num_measurements
        init_val = float(self.hparams.outer.hyperparam_init)

        if c_type == 'scalar':
            c = torch.tensor(init_val)
        elif c_type == 'vector':
            c = torch.ones(m) * init_val
        elif c_type == 'matrix':
            c = torch.eye(m) * init_val
        else:
            raise NotImplementedError("Hyperparameter type not supported")

        return c

    def _get_meta_optimizer(self):
        """
        Initializes the meta optmizer and scheduler.

        Args:
            opt_params: The parameters to optimize over.
                        Type: Tensor.
            hparams: The experiment parameters to use for init.
                    Type: Namespace.
                    Expected to have the consituents:
                        hparams.outer.lr - float, the meta learning rate
                        hparams.outer.lr_decay - [False, float], the exponential decay factor for the learning rate 
                        hparams.outer.optimizer - str in [adam, sgd]
        
        Returns:
            meta_opts: A dictionary containint the meta optimizer and scheduler.
                    If lr_decay is false, the meta_scheduler key has value None.
                    Type: dict.
        """
        opt_type = self.hparams.opt.optimizer
        lr = self.hparams.opt.lr

        if opt_type == 'adam':
            meta_opt = torch.optim.Adam([{'params': self.c}], lr=lr)
        elif opt_type == 'sgd':
            meta_opt = torch.optim.SGD([{'params': self.c}], lr=lr)
        else:
            raise NotImplementedError("Optimizer not supported!")

        if self.hparams.opt.decay:
            meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, self.hparams.opt.lr_decay)
        else:
            meta_scheduler = None

        return meta_opt, meta_scheduler