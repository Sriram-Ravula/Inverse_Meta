from sched import scheduler
import torch
import numpy as np

class GBML(torch.nn.Module):
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args

        self.register_parameter('c', self._init_c())
        self.opt, self.scheduler = self._get_meta_optimizer()
    
    def forward(self, )

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

