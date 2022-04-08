import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

import os
import sys
import torch.utils.tensorboard as tb
import yaml

from algorithms.sgld import SGLD_NCSNv2
from problems import get_forward_operator
from datasets import get_dataset, split_dataset

from utils_new.meta_utils import hessian_vector_product as hvp
from utils_new.meta_loss_utils import meta_loss, get_meta_grad
from utils_new.inner_loss_utils import get_likelihood_grad
from utils_new.metric_utils import Metrics

class GBML(torch.nn.Module):
    def __init__(self, hparams, args):
        super().__init__()
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        #running parameters
        self._init_c()
        self._init_meta_optimizer()
        self._init_dataset()
        self.A = get_forward_operator(self.hparams).to(self.device)

        self.global_iter = 0
        self.best_iter = 0
        self.best_c = None
        self.test_metric = 'psnr'

        #Langevin algorithm
        self.langevin_runner = SGLD_NCSNv2(self.hparams, self.c, self.A).to(self.device)
        if self.hparams.gpu_num == -1:
            self.langevin_runner = torch.nn.DataParallel(self.langevin_runner)

        #logging and metrics
        self.metrics = Metrics(hparams=self.hparams)
        self.log_dir = os.path.join(self.hparams.save_dir, self.args.doc)
        self.image_root = os.path.join(self.log_dir, 'images')
        self.tb_root = os.path.join(self.log_dir, 'tensorboard')

        self._make_log_folder()
        self._save_config()

        self.tb_logger = tb.SummaryWriter(log_dir=self.tb_root)

    def run_meta_opt(self):
        for iter in tqdm(range(self.hparams.opt.num_iters)):
            #TODO checkpointing

            #validate
            if iter % self.hparams.opt.val_iters == 0:
                self._run_validation()
                if not self.hparams.debug:
                    self.metrics.add_metrics_to_tb(self.tb_logger, self.global_iter, "val")
            
            #train
            self._run_outer_step()
            if not self.hparams.debug:
                self.metrics.add_metrics_to_tb(self.tb_logger, self.global_iter, "train")

            self.global_iter += 1

        #use the best c we discovered 
        self.c = self.best_c

        #test
        self._run_test()
        if not self.hparams.debug:
            self.metrics.add_metrics_to_tb(self.tb_logger, self.global_iter, "test")

        #TODO checkpointing

        return

    def _run_outer_step(self):
        meta_grad = 0.0 
        n_samples = 0
        num_batches = self.hparams.opt.batches_per_iter
        if num_batches == -1 or num_batches > len(self.train_loader):
            num_batches = len(self.train_loader)
        elif num_batches == 0:
            return

        for x, x_idx in tqdm(enumerate(self.train_loader)):
            n_samples += x.shape[0]
            
            #(1) Find x(c) by running the inner algorithm
            x = x.to(self.hparams.device)
            y = self.A.forward(self, x, targets=True)

            x_mod = torch.rand_like(x)

            x_hat = self.langevin_runner.forward(x_mod, y)

            #(2) Calculate the meta-gradient and (possibly) update
            #NOTE get meta gradient

            #if we have passed through the requisite number of samples, update
            num_batches -= 1
            if num_batches==0:
                meta_grad /= n_samples
                self._opt_step(meta_grad)

                meta_grad = 0.0 
                n_samples = 0
                num_batches = self.hparams.opt.batches_per_iter
                if num_batches == -1 or num_batches > len(self.train_loader):
                    num_batches = len(self.train_loader)

            #(3) Logging etc.

        return
    
    def forward(self, x, x_mod=None, eval=False):
        #(1) Find x(c) by running the inner optimization
        y = self.A.forward(x)

        if x_mod is None:
            x_mod = torch.rand(x.shape, requires_grad=True).type_as(x)
        
        x_hat = self.langevin_runner.forward(x_mod, y, eval=eval)

        #loss is not reduced over the samples
        with torch.no_grad():
            losses = meta_loss(x_hat=x_hat,
                                x_true=x,
                                reduce_dims=(1,2,3),
                                c = self.c,
                                measurement_loss=self.hparams.outer.measurement_loss,
                                meta_loss_type=self.hparams.outer.meta_loss_type,
                                reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale,
                                ROI_loss=self.hparams.outer.ROI_loss,
                                ROI=self.hparams.outer.ROI)

        if not eval:
            if self.hparams.outer.meta_type == 'mle':
                meta_grad = self.mle_step(x_hat, x, y)
            else: 
                raise NotImplementedError("Only MLE learning implemented!")
        else:
            meta_grad = None

        return x_hat.detach(), losses, meta_grad
    
    def _opt_step(self, meta_grad):
        """
        Will take an optimization step (and scheduler if applicable). 
        Sets c.grad to True then False.
        """
        self.opt.zero_grad()

        self.c.requires_grad_()
        if type(self.c.grad) == type(None): #dummy update to make sure grad is initialized
            dummy_loss = torch.sum(self.c)
            dummy_loss.backward()
        self.c.grad.copy_(meta_grad)
        self.opt.step()
        self.c.requires_grad_(False)

        self.c.clamp_(min=0.)
    
        if self.scheduler is not None and not self.hparams.outer.decay_on_val:
            self.scheduler.step()

    def _mle_grad(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an MLE step.
        grad_c(meta_loss) = - grad_x_c[recon_loss] * (meta_loss)
        (1) Find meta loss
        (2) HVP of grad_x_c(recon_loss) * grad_x(meta_loss)

        Sets c.grad to True then False.
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

        self.c.requires_grad_(False)

        return out_grad
    
    def _init_dataset(self):
        _, base_dataset = get_dataset(self.hparams)
        split_dict = split_dataset(base_dataset, self.hparams)
        train_dataset = split_dict['train']
        val_dataset = split_dict['val']
        test_dataset = split_dict['test']

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=1, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.test_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)

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

        self.c = c.to(self.device)

    def _init_meta_optimizer(self):
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

        self.opt =  meta_opt
        self.scheduler = meta_scheduler

    def _make_log_folder(self):
        if os.path.exists(self.log_dir):
            sys.exit("Folder exists. Program halted.")
        else:
            os.makedirs(self.log_dir)
            os.makedirs(self.image_root)
            os.makedirs(self.tb_root)
    
    def _save_config(self):
        with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False)

    def _add_tb_images(self, images, tag):
        grid_img = torchvision.utils.make_grid(images.cpu(), nrow=images.shape[0]//2)
        self.tb_logger.add_image(tag, grid_img, global_step=self.global_iter)
