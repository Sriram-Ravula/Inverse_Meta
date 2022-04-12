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

from utils_new.exp_utils import save_images
from utils_new.meta_utils import hessian_vector_product as hvp
from utils_new.meta_loss_utils import meta_loss, get_meta_grad
from utils_new.inner_loss_utils import get_likelihood_grad, log_cond_likelihood_loss
from utils_new.metric_utils import Metrics

#TODO then fancy stuff like saving initialisations, different meta losses, different inner algo
#       then ROI etc.

class GBML:
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        #running parameters
        self._init_c()
        self._init_meta_optimizer()
        self._init_dataset()
        self.A = get_forward_operator(self.hparams).to(self.device)

        self.global_epoch = 0
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
                self._add_metrics_to_tb("val")
            
            #train
            self._run_outer_step()
            self._add_metrics_to_tb("train")

            self.global_epoch += 1

        #use the best c we discovered 
        if self.hparams.gpu_num != -1:
            self.langevin_runner.set_c(self.best_c)
        else:
            self.langevin_runner.module.set_c(self.best_c)

        #test
        self._run_test()
        self._add_metrics_to_tb("test")

        #TODO checkpointing

    def _run_outer_step(self):
        """
        Runs one epoch of meta learning training.
        """
        self._print_if_verbose("\nTRAINING\n")

        meta_grad = 0.0 
        n_samples = 0
        num_batches = self.hparams.opt.batches_per_iter
        if num_batches < 0 or num_batches > len(self.train_loader):
            num_batches = len(self.train_loader)
        elif num_batches == 0:
            return

        for i, (x, x_idx) in tqdm(enumerate(self.train_loader)):
            x_hat, x, y = self._shared_step(x, "train")

            #(2) Calculate the meta-gradient and (possibly) update
            meta_grad += self._mle_grad(x_hat, x, y)
            
            #if we have passed through the requisite number of samples, update
            n_samples += x.shape[0]
            num_batches -= 1
            if num_batches==0:
                meta_grad /= n_samples
                self._opt_step(meta_grad)

                if self.hparams.gpu_num != -1:
                    self.langevin_runner.set_c(self.c)
                else:
                    self.langevin_runner.module.set_c(self.c)

                meta_grad = 0.0 
                n_samples = 0
                num_batches = self.hparams.opt.batches_per_iter
                if num_batches < 0 or num_batches > len(self.train_loader):
                    num_batches = len(self.train_loader)

            #logging and saving
            if i == 0:
                self._save_all_images(x_hat, x, y, x_idx, "train")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "train", False)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "train"), "\n")

    def _run_validation(self):
        """
        Runs one epoch of meta learning validation.
        """
        self._print_if_verbose("\nVALIDATING\n")

        for i, (x, x_idx) in tqdm(enumerate(self.val_loader)):
            x_hat, x, y = self._shared_step(x, "val")

            #logging and saving
            if i == 0:
                self._save_all_images(x_hat, x, y, x_idx, "val")

        new_best_dict = self.metrics.aggregate_iter_metrics(self.global_epoch, "val", True)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "val"), "\n")

        #aggregate metrics, see if we have a new best
        #if we do, set the new best c. If we don't, decay lr (if applicable)
        if self.global_epoch > 0:
            if new_best_dict is not None and self.test_metric in new_best_dict:
                self._print_if_verbose("\nNEW BEST VAL " + self.test_metric + ": ", new_best_dict[self.test_metric], "\n")
                self.best_c = self.c.detach().clone()
            elif self.scheduler is not None and self.hparams.outer.decay_on_val:
                LR_OLD = self.opt.param_groups[0]['lr']
                self.scheduler.step()
                LR_NEW = self.opt.param_groups[0]['lr']
                self._print_if_verbose("\nVAL LOSS HASN'T IMPORVED; DECAYING LR: ", LR_OLD, " --> ", LR_NEW)
    
    def _run_test(self):
        self._print_if_verbose("\nTESTING\n")

        for i, (x, x_idx) in tqdm(enumerate(self.test_loader)):
            x_hat, x, y = self._shared_step(x, "test")

            #logging and saving
            self._save_all_images(x_hat, x, y, x_idx, "test")
        
        self.metrics.aggregate_iter_metrics(self.global_epoch, "test", False)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")
    
    def _shared_step(self, x, iter_type):
        """
        given a batch of samples x, solve the inverse problem and log the batch metrics
        """
        #Find x(c) by running the inner algorithm
        x = x.to(self.hparams.device)
        with torch.no_grad():
            y = self.A(x, targets=True)

        x_mod = torch.rand_like(x)

        x_hat = self.langevin_runner(x_mod, y)

        #logging
        self._add_batch_metrics(x_hat, x, y, iter_type)

        return x_hat, x, y
    
    @torch.no_grad()
    def _add_batch_metrics(self, x_hat, x, y, iter_type):
        """
        Adds metrics for a single batch to the metrics object.
        """
        real_meas_loss = log_cond_likelihood_loss(torch.tensor(1.), y, self.A, x_hat, reduce_dims=(1)) #get element-wise ||Ax - y||^2 (i.e. sse for each sample)
        weighted_meas_loss = log_cond_likelihood_loss(self.c, y, self.A, x_hat, reduce_dims=(1)) #get element-wise C||Ax - y||^2 (i.e. sse for each sample)
        all_meta_losses = meta_loss(x_hat, x, (1,2,3), self.c, 
                                    meta_loss_type=self.hparams.outer.meta_loss_type,
                                    reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                    reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                    reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale)

        extra_metrics_dict = {"real_meas_loss": real_meas_loss.cpu().numpy().flatten(),
                                "weighted_meas_loss": weighted_meas_loss.cpu().numpy().flatten(),
                                "meta_loss_"+str(self.hparams.outer.meta_loss_type): all_meta_losses[0].cpu().numpy().flatten(),
                                "meta_loss_reg": all_meta_losses[1].cpu().numpy().flatten(),
                                "meta_loss_total": all_meta_losses[2].cpu().numpy().flatten()}

        self.metrics.add_external_metrics(extra_metrics_dict, self.global_epoch, iter_type)
        self.metrics.calc_iter_metrics(x_hat, x, self.global_epoch, iter_type)

    def _save_all_images(self, x_hat, x, y, x_idx, iter_type):
        """
        Given true, measurement, and recovered images, save to tensorboard and png.
        """
        if self.hparams.debug or (not self.hparams.save_imgs):
            return

        meas_images = self.A.get_measurements_image(x, targets=True)

        true_path = os.path.join(self.image_root, iter_type)
        meas_path = os.path.join(self.image_root, iter_type + "_meas")
        if iter_type == "test":
            recovered_path = os.path.join(self.image_root, iter_type + "_recon")
        else:
            recovered_path = os.path.join(self.image_root, iter_type + "_recon", "epoch_"+str(self.global_epoch))
        
        self._add_tb_images(x_hat, "recovered " + iter_type + " images")
        os.makedirs(recovered_path)
        self._save_images(x_hat, x_idx, recovered_path)

        if iter_type == "test" or self.global_epoch == 0:
            self._add_tb_images(x, iter_type + " images")
            os.makedirs(true_path)
            self._save_images(x, x_idx, true_path)

            if meas_images is not None:
                self._add_tb_images(meas_images, iter_type + " measurements")
                os.makedirs(meas_path)
                self._save_images(meas_images, x_idx, meas_path)
    
    def _opt_step(self, meta_grad):
        """
        Will take an optimization step (and scheduler if applicable). 
        Sets c.grad to True then False.
        """
        self.opt.zero_grad()

        self.c.requires_grad_()

         #dummy update to make sure grad is initialized
        if type(self.c.grad) == type(None):
            dummy_loss = torch.sum(self.c)
            dummy_loss.backward()

        self.c.grad.copy_(meta_grad)
        self.opt.step()

        self.c.requires_grad_(False)

        self.c.clamp_(min=0.)
    
        if self.scheduler is not None and not self.hparams.outer.decay_on_val:
            LR_OLD = self.opt.param_groups[0]['lr']
            self.scheduler.step()
            LR_NEW = self.opt.param_groups[0]['lr']
            self._print_if_verbose("\nDECAYING LR: ", LR_OLD, " --> ", LR_NEW)

    def _mle_grad(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an MLE step.
        grad_c(meta_loss) = - grad_x_c[recon_loss] * (meta_loss)
        (1) Find meta loss
        (2) Get the HVP grad_x_c(recon_loss) * grad_x(meta_loss)
        """
        #(1) Get gradients of Meta loss w.r.t. image and hyperparams
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
        cond_log_grad = get_likelihood_grad(self.c, y, self.A, x_hat, self.hparams.use_autograd, 
                                            exp_params=self.hparams.outer.exp_params, retain_graph=True, 
                                            create_graph=True) #gradient of likelihood loss (with hyperparam) w.r.t image
        
        out_grad = 0.0
        out_grad -= hvp(self.c, cond_log_grad, grad_x_meta_loss)
        out_grad += grad_c_meta_loss

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
        if not self.hparams.debug:
            if os.path.exists(self.log_dir):
                sys.exit("Folder exists. Program halted.")
            else:
                os.makedirs(self.log_dir)
                os.makedirs(self.image_root)
                os.makedirs(self.tb_root)
    
    def _save_config(self):
        if not self.hparams.debug:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
                yaml.dump(self.hparams, f, default_flow_style=False)

    def _add_tb_images(self, images, tag):
        if not self.hparams.debug and self.hparams.save_imgs:
            grid_img = torchvision.utils.make_grid(images.cpu(), nrow=images.shape[0]//2)
            self.tb_logger.add_image(tag, grid_img, global_step=self.global_epoch)
    
    def _add_metrics_to_tb(self, iter_type):
        if not self.hparams.debug:
            self.metrics.add_metrics_to_tb(self.tb_logger, self.global_epoch, iter_type)

    def _save_images(self, images, img_indices, save_path):
        if not self.hparams.debug and self.hparams.save_imgs:
            save_images(images, img_indices, save_path)

    def _print_if_verbose(self, *text):
        if self.hparams.verbose:
            print("".join(str(t) for t in text))