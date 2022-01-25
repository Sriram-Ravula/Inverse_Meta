from sys import path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from time import time
import os

from utils.utils import dict2namespace, split_dataset, init_c, get_meta_optimizer, plot_images, get_measurement_images
from utils.loss_utils import get_A, get_measurements, get_likelihood_grad, get_meta_grad, get_loss_dict, meta_loss
from utils.alg_utils import SGLD_inverse, hessian_vector_product, Ax, cg_solver, SGLD_inverse_eval
from utils.metrics_utils import Metrics

from ncsnv2.models.ncsnv2 import NCSNv2, NCSNv2Deepest
from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.datasets import get_dataset


class MetaLearner:
    """
    Meta Learning for inverse problems
    """
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.__init_net()
        self.__init_datasets()
        self.__init_problem()
        self.metrics = Metrics(hparams=hparams)
        if not self.hparams.outer.debug:
            from utils.logging_utils import Logger
            self.logger = Logger(self.metrics, self, hparams, os.path.join(hparams.save_dir, args.doc))
        return
    
    def __init_net(self):
        """Initializes score net and related attributes"""
        if self.hparams.net.model != "ncsnv2":
            raise NotImplementedError 
        
        if self.hparams.outer.verbose:
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

        if net_config.model.ema:
            ema_helper = EMAHelper(mu=net_config.model.ema_rate)
            ema_helper.register(test_score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(test_score)

        test_score.eval()
        for param in test_score.parameters():
            param.requires_grad = False

        self.model = test_score
        self.sigmas = get_sigmas(net_config).cpu()
        self.model_config = net_config

        if self.hparams.outer.verbose:
            end = time()
            print("\nNET TIME: ", str(end - start), "S\n")

        return 
    
    def __init_datasets(self):
        if self.hparams.outer.verbose:
            print("\nINITIALIZING DATA\n")
            start = time()

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default='exp', help='Path where data is located')
        args = parser.parse_args(["--data_path", self.hparams.data.data_path])

        _, base_dataset = get_dataset(args, self.model_config)

        train_dataset, val_dataset, test_dataset = split_dataset(base_dataset, self.hparams)

        if self.hparams.outer.use_validation:
            self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                        num_workers=2, drop_last=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=2, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=2, drop_last=True)

        if self.hparams.outer.verbose:
            end = time()
            print("\nDATA TIME: ", str(end - start), "S\n")

        return
    
    def __init_problem(self):
        if self.hparams.outer.verbose:
            print("\nINITIALIZING INNER PROBLEM\n")
            start = time()

        self.A = get_A(self.hparams)
        self.noisy = self.hparams.problem.add_noise

        if self.A is not None:
            self.A = self.A.to(self.hparams.device)

        self.c = init_c(self.hparams).to(self.hparams.device)

        if self.hparams.outer.lr_decay and self.hparams.outer.optimizer != 'linesearch':
            self.meta_opt, self.meta_scheduler = get_meta_optimizer(self.c, self.hparams)
        else:
            self.meta_opt = get_meta_optimizer(self.c, self.hparams)

        #values used for loss min appx
        s_idx = len(self.sigmas)-1
        self.loss_scale = 1 / (self.sigmas[s_idx]**2)
        self.labels = torch.ones(self.hparams.data.train_batch_size, device=self.hparams.device) * s_idx
        self.labels = self.labels.long()

        if self.hparams.outer.use_autograd and self.hparams.problem.measurement_type == 'inpaint':
            self.efficient_inp = True
        else:
            self.efficient_inp = False

        self.global_iter = 0
        self.best_iter = 0

        self.grad_norms = []
        self.grads = []
        self.c_list = [self.c.detach().cpu()]

        self.ROI = self.hparams.outer.ROI 
        if self.hparams.outer.ROI:
            self.val_metric = 'roi_nmse'
        else:
            self.val_metric = 'nmse' 

        self.save_inits = self.hparams.outer.save_inits
        if self.save_inits:
            self.x_inits = {}

        if self.hparams.outer.verbose:
            end = time()
            print("\nPROBLEM TIME: ", str(end - start), "S\n")

        return

    def __load_inits(self, indices):
        """"Method for loading saved initializations"""
        if self.hparams.outer.verbose:
            print("\nLOADING SAVED INITIALIZATONS\n")

        out_x = None

        for i in indices:
            if str(i) not in self.x_inits:
                self.x_inits[str(i)] = torch.rand(self.hparams.data.image_shape)
            
            if out_x is None:
                out_x = self.x_inits[str(i)].clone().unsqueeze(0)
            else:
                out_x = torch.cat((out_x, self.x_inits[str(i)].unsqueeze(0)), 0)
        
        if len(out_x.shape) == 3:
            out_x = out_x.unsqueeze(0)
        
        return out_x.to(self.hparams.device).requires_grad_()
    
    def __save_inits(self, x_out, indices):
        """Method for saving initializations at the end of a training loop"""
        if self.hparams.outer.verbose:
            print("\SAVING INITIALIZATONS\n")

        for x_i, i in enumerate(indices):
            self.x_inits[str(i)] = x_out[x_i].detach().cpu().clone()

        return
    
    def run_meta_opt(self):
        for iter in tqdm(range(self.hparams.outer.num_iters)):
            #checkpointing
            if not self.hparams.outer.debug and iter % self.hparams.outer.checkpoint_iters == 0:
                self.logger.checkpoint()
            #val
            if self.hparams.outer.use_validation and iter % self.hparams.outer.val_iters == 0:
                self.run_validation()
                if not self.hparams.outer.debug:
                    self.logger.add_metrics_to_tb('val')            
            #train
            self.run_outer_step()
            if not self.hparams.outer.debug:
                self.logger.add_metrics_to_tb('train')      
                 
            self.global_iter += 1
        
        #replace current c with the one from the best iteration
        if self.hparams.outer.use_validation:
            self.c.copy_(self.c_list[self.best_iter])

        #Test
        if self.hparams.outer.verbose:
            print("\nBEGINNING TESTING\n")
            
        self.val_or_test(validate=False) 
        if not self.hparams.outer.debug:
            self.logger.add_metrics_to_tb('test') 

        if self.hparams.outer.verbose:
            print("\nTEST LOSS: ", self.metrics.get_metric(self.global_iter, 'test', self.val_metric), '\n')
            print('\n', self.metrics.get_all_metrics(self.global_iter, 'test'), '\n')
        
        if not self.hparams.outer.debug:
            self.logger.checkpoint()

        return
    
    def run_outer_step(self):
        if self.hparams.outer.optimizer != 'linesearch':
            self.meta_opt.zero_grad()

        if self.hparams.outer.optimizer == 'linesearch':
            meta_grad, x_hat_out, x_out, y_out = self.outer_step()
        else: 
            meta_grad = self.outer_step()

        if self.hparams.outer.optimizer != 'linesearch':
            self.c.requires_grad_()
            if type(self.c.grad) == type(None): #dummy update to make sure grad is initialized
                dummy_loss = torch.sum(self.c)
                dummy_loss.backward()
            self.c.grad.copy_(meta_grad)
            self.meta_opt.step()
            self.c.requires_grad_(False)
        else:
            stepsize = self.line_search(meta_grad, x_hat_out, x_out, y_out)
            self.c = self.c - stepsize * meta_grad

        if self.hparams.outer.optimizer != 'linesearch' and self.hparams.outer.lr_decay and not self.hparams.outer.use_validation:
            self.meta_scheduler.step()
            print("\nDECAYING LR\n")

        self.grads.append(meta_grad.detach().cpu())
        self.grad_norms.append(torch.norm(meta_grad.flatten()).item())
        self.c_list.append(self.c.detach().cpu())

        if self.hparams.outer.verbose:
            print("\nTRAIN LOSS: ", self.metrics.get_metric(self.global_iter, 'train', self.val_metric), '\n')
            print('\n', self.metrics.get_all_metrics(self.global_iter, 'train'), '\n')
            print("\nGRADIENT NORM: ", self.grad_norms[-1], '\n')
            print("\nC MEAN: ", torch.mean(self.c_list[-1]), '\n')
            print("\nC STD: ", torch.std(self.c_list[-1]), '\n')
            print("\nC MIN: ", torch.min(self.c_list[-1]), '\n')
            print("\nC MAX: ", torch.max(self.c_list[-1]), '\n')

        return 
    
    def outer_step(self):  
        meta_grad = 0.0 
        n_samples = 0
        num_batches = self.hparams.outer.batches_per_iter

        if self.hparams.outer.optimizer == 'linesearch':
            x_hat_out = None
            x_out = None
            y_out = None

        for i, (x, x_idx) in tqdm(enumerate(self.train_loader)):
            if num_batches == 0:
                break

            n_samples += x.shape[0]

            x_idx = x_idx.cpu().numpy().flatten()

            if self.hparams.outer.meta_type == 'maml':
                self.c.requires_grad_()
            
            #(1) Find x(c) by running the inner optimization
            x = x.to(self.hparams.device)
            y = get_measurements(self.A, x, self.hparams, self.efficient_inp, noisy=self.noisy)

            if self.save_inits:
                x_mod = self.__load_inits(x_idx)
            else:
                x_mod = torch.rand(x.shape, device=self.hparams.device, requires_grad=True)
            x_hat = SGLD_inverse(self.c, y, self.A, x_mod, self.model, self.sigmas, self.hparams, self.efficient_inp)
            
            #(2) Find meta gradient
            if self.hparams.outer.meta_type == 'maml':
                meta_grad += self.maml_step(x_hat, x)
            elif self.hparams.outer.meta_type == 'implicit':
                meta_grad += self.implicit_maml_step(x_hat, x, y)
            elif self.hparams.outer.meta_type == 'mle':
                meta_grad += self.mle_step(x_hat, x, y)
            else: 
                raise NotImplementedError
            
            x_hat.requires_grad_(False)
            self.c.requires_grad_(False)

            if self.save_inits:
                self.__save_inits(x_hat, x_idx)

            if self.hparams.outer.optimizer == 'linesearch':
                if x_hat_out is None:
                    x_hat_out = x_hat.detach().cpu().clone()
                else:
                    x_hat_out = torch.cat((x_hat_out, x_hat.detach().cpu().clone()), 0)
                
                if x_out is None:
                    x_out = x.detach().cpu().clone()
                else:
                    x_out = torch.cat((x_out, x.detach().cpu().clone()), 0)
                
                if y_out is None:
                    y_out = y.detach().cpu().clone()
                else:
                    y_out = torch.cat((y_out, y.detach().cpu().clone()), 0)

            loss_metrics = get_loss_dict(y, self.A, x_hat, x, self.hparams, self.efficient_inp)
            self.metrics.calc_iter_metrics(x_hat, x, self.global_iter, 'train')
            self.metrics.add_external_metrics(loss_metrics, self.global_iter, 'train')

            #just plot in tensorboard for now - this isnt working
            # if not self.hparams.outer.debug and self.hparams.outer.plot_imgs:
            #     plot_images(x, "Training Images")
            #     plot_images(get_measurement_images(x, self.hparams), "Training Measurements")
            #     plot_images(x_hat, "Reconstructed")

            num_batches -= 1

        self.metrics.aggregate_iter_metrics(self.global_iter, 'train', return_best=False)
        
        meta_grad /= n_samples

        if self.hparams.outer.optimizer == 'linesearch':
            return meta_grad, x_hat_out, x_out, y_out
        else:
            return meta_grad

    def maml_step(self, x_hat, x):
        """
        Calculates the meta-gradient for a MAML step. 
        grad_c(x_hat) = grad_c(x_hat_0)*...*grad_c(x_hat_(T-1))*grad_(x_hat_(T-1))(x_hat) + grad_c(x_hat)
        (1) Find the meta loss 
        (2) Find the HVP of grad_c(x_hat)*(meta_loss)
        """
        #(1)
        grad_x_meta_loss = get_meta_grad(x_hat, x, self.hparams, retain_graph=True)

        #(2)
        out_grad = hessian_vector_product(self.c, x_hat, grad_x_meta_loss, self.hparams)

        return out_grad 
    
    def implicit_maml_step(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an Implicit MAML step.
        grad_c(x_hat) = - grad_x_c[recon_loss] * grad^2_x[recon_loss]^-1 * (meta_loss)
        (1) Find meta loss
        (2) Do conjugate gradient and find the inverse HVP
        (3) Find the HVP of grad_x_c[recon_loss]*ihvp
        """
        #(1)
        grad_x_meta_loss = get_meta_grad(x_hat, x, self.hparams)

        #(2)
        cond_log_grad = get_likelihood_grad(self.c, y, self.A, x_hat, self.hparams, self.loss_scale, \
            efficient_inp=self.efficient_inp, retain_graph=True, create_graph=True)  
        prior_grad = self.model(x_hat, self.labels) 

        hvp_helper = Ax(x_hat, (cond_log_grad - prior_grad), self.hparams, retain_graph=True)

        ihvp = cg_solver(hvp_helper, grad_x_meta_loss, self.hparams)

        #(3)
        self.c.requires_grad_()

        cond_log_grad = get_likelihood_grad(self.c, y, self.A, x_hat, self.hparams, self.loss_scale, \
            efficient_inp=self.efficient_inp, retain_graph=True, create_graph=True)  
        
        out_grad = -hessian_vector_product(self.c, cond_log_grad, ihvp, self.hparams)

        return out_grad

    def mle_step(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an MLE step.
        grad_c(x_hat) = - grad_x_c[recon_loss] * (meta_loss)
        (1) Find meta loss
        (2) HVP of grad_x_c(recon_loss) * grad_x(meta_loss)
        """
        #(1)
        grad_x_meta_loss = get_meta_grad(x_hat, x, self.hparams)
        
        #(2)
        self.c.requires_grad_()
        
        cond_log_grad = get_likelihood_grad(self.c, y, self.A, x_hat, self.hparams, self.loss_scale, \
            efficient_inp=self.efficient_inp, retain_graph=True, create_graph=True) 
        
        out_grad = -hessian_vector_product(self.c, cond_log_grad, grad_x_meta_loss, self.hparams)

        return out_grad
    
    def run_validation(self):
        """
        Runs a validation epoch and checks if we have a new best loss
        """
        if self.hparams.outer.verbose:
            print("\nBEGINNING VALIDATION\n")
        new_best_dict = self.val_or_test(validate=True)

        #check if we have a new best validation loss
        #if so, update best iter
        if self.global_iter > 0:
            if new_best_dict is not None and self.val_metric in new_best_dict:
                best_value = new_best_dict[self.val_metric]
                self.best_iter = self.global_iter
                if self.hparams.outer.verbose:
                    print("\nNEW BEST VAL LOSS: ", best_value, "\n")
            elif self.hparams.outer.optimizer != 'linesearch' and self.hparams.outer.lr_decay:
                self.meta_scheduler.step()
                if self.hparams.outer.verbose :
                    print("\nVAL LOSS HASN'T IMPROVED; DECAYING LR\n")

        if self.hparams.outer.verbose:
            print("\nVAL LOSS: ", self.metrics.get_metric(self.global_iter, 'val', self.val_metric), '\n')
            print('\n', self.metrics.get_all_metrics(self.global_iter, 'val'), '\n')
        
        return

    def val_or_test(self, validate):
        """
        Performs one validation or test run. Calculates and stores related metrics.

        Returns:
            new_best_dict: dict or None. A dictionary with any new best val or test metrics and their values. 
                           If no new bests are found, returns None.
        """
        if validate:
            cur_loader = self.val_loader
            iter_type = 'val'
        else:
            cur_loader = self.test_loader
            iter_type = 'test'

        for i, (x, x_idx) in tqdm(enumerate(cur_loader)):
            x_idx = x_idx.cpu().numpy().flatten()
            x = x.to(self.hparams.device)
            y = get_measurements(self.A, x, self.hparams, self.efficient_inp, noisy=self.noisy)

            x_mod = torch.rand(x.shape, device=self.hparams.device)
            x_hat = SGLD_inverse_eval(self.c, y, self.A, x_mod, self.model, self.sigmas, self.hparams, self.efficient_inp)

            loss_metrics = get_loss_dict(y, self.A, x_hat, x, self.hparams, self.efficient_inp)
            self.metrics.calc_iter_metrics(x_hat, x, self.global_iter, iter_type)
            self.metrics.add_external_metrics(loss_metrics, self.global_iter, iter_type)

            if not self.hparams.outer.debug and self.hparams.outer.plot_imgs:
                # plot_images(x, "True Images")
                # plot_images(get_measurement_images(x, self.hparams), "Measurements")
                # plot_images(x_hat, "Reconstructed")
                if self.global_iter == 0 or not validate:
                    self.logger.add_tb_images(x, iter_type + "_imgs_" + str(self.global_iter))
                    self.logger.add_tb_measurement_images(x, iter_type + "_meas_" + str(self.global_iter))
                    self.logger.save_images(x, x_idx, iter_type + "_imgs_" + str(self.global_iter))
                    self.logger.save_image_measurements(x, x_idx, iter_type + "_imgs_meas_" + str(self.global_iter))
                self.logger.add_tb_images(x_hat, iter_type + "_recons_" + str(self.global_iter))
                self.logger.save_images(x_hat, x_idx, iter_type + "_recons_" + str(self.global_iter))
            
            if not self.hparams.outer.debug and not validate:
                self.logger.save_images(x, x_idx, "Test_true")
                self.logger.save_image_measurements(x, x_idx, "Test_meas", self.noisy)
                self.logger.save_images(x_hat, x_idx, "Test_recon")
        
        if validate:
            new_best_dict = self.metrics.aggregate_iter_metrics(self.global_iter, iter_type, return_best=True)
            return new_best_dict
        else:
            self.metrics.aggregate_iter_metrics(self.global_iter, iter_type, return_best=False)
            return

    def line_search(self, meta_grad, x_hat, x, y):
        """Performs a step of line search at the end of an iteration"""
        if self.hparams.outer.verbose:
            print("\nPERFORMING LINE SEARCH\n")

        meta_grad = meta_grad.to(self.hparams.device)
        x_hat = x_hat.to(self.hparams.device)
        x = x.to(self.hparams.device)
        y = y.to(self.hparams.device)

        s_idx = len(self.sigmas)-1
        labels = torch.ones(x_hat.shape[0], device=self.hparams.device) * s_idx
        labels = labels.long()

        t = 1
        alpha = 1e-4
        beta = 0.5
        stop_flag = False
        
        while not stop_flag:
            #(1) c + t*delta(c)
            with torch.no_grad():
                c_new = self.c - t * meta_grad

            #(2) grad_x recon_loss(x_hat(c), c + t*delta_c)
            cond_log_grad = get_likelihood_grad(c_new, y, self.A, x_hat, self.hparams, self.loss_scale, \
                efficient_inp=self.efficient_inp, retain_graph=False, create_graph=False) 

            with torch.no_grad():
                prior_grad = self.model(x_hat, labels) 
            
                #x_hat(c + t*delta_c) = x_hat(c) - grad_x recon_loss(x_hat(c), c + t*delta_c)
                x_hat_new = x_hat + prior_grad - cond_log_grad
            
                #F(c + t*delta_c) = meta_loss(x_hat(c + t*delta_c))
                f_new = meta_loss(x_hat_new, x, self.hparams)

                #F(c) = meta_loss(x_hat(c))
                f_old = meta_loss(x_hat, x, self.hparams)

                #grad_c F(x_hat(c))^T (- grad_c F(x_hat(c))) = ||grad_c F(x_hat(c))||^2
                grad_dir = torch.sum(meta_grad ** 2)

                LHS_RHS_resid = f_new - (f_old - alpha * t * grad_dir)

                #check if LHS < RHS
                if LHS_RHS_resid > 0:
                    if self.hparams.outer.verbose:
                        print("\nRESIDUAL TOO HIGH: ", str(LHS_RHS_resid.item()), " - DECAYING LR\n")
                    t = beta * t
                else:
                    if self.hparams.outer.verbose:
                        print("\nLR FOUND: ", str(t), " - STOPPING LINESEARCH")
                    stop_flag = True
        
        return t
        