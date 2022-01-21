from cupshelpers import Printer
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from time import time

from utils import dict2namespace, split_dataset, init_c, get_meta_optimizer
from loss_utils import get_A, meta_loss, get_measurements, gradient_log_cond_likelihood, log_cond_likelihood_loss
from alg_utils import SGLD_inverse, hessian_vector_product, Ax, cg_solver

from ncsnv2.models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.datasets import get_dataset, utils


class MetaLearner:
    """
    Meta Learning for inverse problems
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.__init_net()
        self.__init_datasets()
        self.__init_problem()
        return
    
    def __init_net(self):
        if self.hparams.net.model != "ncsnv2":
            raise NotImplementedError #TODO implement other models!
        
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
            print("\nNET TIME: ", str(end - start), " S\n")

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

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=2, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=2, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=2, drop_last=True)

        if self.hparams.outer.verbose:
            end = time()
            print("\nDATA TIME: ", str(end - start), " S\n")

        return
    
    def __init_problem(self):
        if self.hparams.outer.verbose:
            print("\nINITIALIZING INNER PROBLEM\n")
            start = time()

        self.A = get_A(self.hparams)
        if self.A is not None:
            self.A = self.A.to(self.hparams.device)

        self.c = utils.init_c(self.hparams).to(self.hparams.device)

        if self.hparams.outer.lr_decay:
            self.meta_opt, self.meta_scheduler = utils.get_meta_optimizer(self.c, self.hparams.outer.lr_decay)
        else:
            self.meta_opt = utils.get_meta_optimizer(self.c, self.hparams.outer.lr_decay)
        
        #values used for loss min appx
        s_idx = len(self.sigmas)-1
        self.loss_scale = 1 / (self.sigmas[s_idx]**2)
        self.labels = torch.ones(self.hparams.data.train_batch_size, device=self.hparams.device) * s_idx
        self.labels = self.labels.long()

        if self.hparams.outer.auto_cond_log and self.hparams.outer.hyperparam_type == 'inpaint':
            self.efficient_inp = True
        else:
            self.efficient_inp = False

        self.global_iter = 0
        self.best_iter = 0
        self.best_val_iter = 0 #used to match best_iter and val_idx if val_iters is not = 1

        self.meta_losses = []
        self.val_losses = []
        self.test_losses = []
        self.grad_norms = []
        self.grads = []
        self.c_list = [self.c.detach().cpu()]

        if self.hparams.outer.verbose:
            end = time()
            print("\nPROBLEM TIME: ", str(end - start), " S\n")

        return
    
    def run_meta_opt(self):
        for iter in tqdm(range(self.hparams.outer.num_iters)):
            #validate
            if iter % self.hparams.val_iters == 0:
                self.val_or_test(validate=True)
            
            #train
            meta_grad, meta_train_loss = self.outer_step()

            self.meta_losses.append(meta_train_loss)
            self.grads.append(meta_grad)
            self.grad_norms.append(torch.norm(meta_grad.flatten()).item())
            self.c_list.append(self.c.detach().cpu())

            if self.hparams.outer.verbose:
                print("\nTRAIN META LOSS: ", meta_train_loss)
                print("\nGRADIENT NORM: ", self.grad_norms[-1], '\n')

            self.global_iter += 1
        
        #Test
        self.val_or_test(validate=False)

        if self.hparams.outer.verbose:
            print("FINISHED!\n")
            print("TEST LOSSES: ")
            print(self.test_losses)
            print("VAL LOSSES: ")
            print(self.val_losses)
            print("META LOSSES: ")
            print(self.meta_losses)
            print("GRAD NORMS: ")
            print(self.grad_norms)

        return
    
    def outer_step(self):  
        self.meta_opt.zero_grad()
        meta_grad = 0.0 
        cur_meta_loss = 0.0
        n_samples = 0
        num_batches = self.hparams.outer.batches_per_iter

        for i, (x, _) in tqdm(enumerate(self.train_loader)):
            if num_batches == 0:
                break
            n_samples += x.shape[0]

            if self.hparams.outer.meta_type == 'maml':
                self.c.requires_grad_()
            
            #(1) Find x(c)
            x = x.to(self.hparams.device)
            y = get_measurements(self.A, x, self.hparams, self.efficient_inp)

            x_mod = torch.rand(x.shape, device=self.hparams.device, requires_grad=True)
            x_hat = SGLD_inverse(self.c, y, self.A, x_mod, self.model, self.sigmas, self.hparams)

            with torch.no_grad():
                cur_meta_loss += meta_loss(x_hat, x, self.hparams).item()
            
            #(2) Find meta loss
            if self.hparams.outer.meta_type == 'maml':
                #simply find the HVP grad_c(x)*grad_x(meta_loss)
                grad_x_meta_loss = torch.autograd.grad(meta_loss(x_hat, x, self.hparams), x_hat, retain_graph=True)[0]
                meta_grad += hessian_vector_product(self.c, x_hat, grad_x_meta_loss, self.hparams)

            elif self.hparams.outer.meta_type == 'implicit':
                #(2)a - do conjugate gradient and find the inverse HVP ihvp=[grad^2_x(inner_loss)]^-1 * grad_x(meta_loss)
                grad_x_meta_loss = torch.autograd.grad(meta_loss(x_hat, x, self.hparams), x_hat)[0]

                if self.hparams.outer.auto_cond_log:
                    cond_log_grad = torch.autograd.grad(log_cond_likelihood_loss\
                        (self.c, y, self.A, x_hat, self.hparams, self.loss_scale, self.efficient_inp), x_hat, create_graph=True)[0]
                else:
                    cond_log_grad = gradient_log_cond_likelihood(self.c, y, self.A, x_hat, self.hparams, self.loss_scale)  
                prior_grad = self.model(x_hat, self.labels) 

                hvp_helper = Ax(x_hat, (cond_log_grad - prior_grad), self.hparams, retain_graph=True)

                ihvp = cg_solver(hvp_helper, grad_x_meta_loss, self.hparams)

                #(2)b Find the HVP of grad_x_c(inner_loss)*ihvp
                self.c.requires_grad_()
                if self.hparams.outer.auto_cond_log:
                    cond_log_grad = torch.autograd.grad(log_cond_likelihood_loss\
                        (self.c, y, self.A, x_hat, self.hparams, self.loss_scale, self.efficient_inp), x_hat, create_graph=True)[0]
                else:
                    cond_log_grad = gradient_log_cond_likelihood(self.c, y, self.A, x_hat, self.hparams, self.loss_scale)  
            
                meta_grad -= hessian_vector_product(self.c, cond_log_grad, ihvp, self.hparams)

            elif self.hparams.outer.meta_type == 'mle':
                #We just want to find the HVP grad_x_c(inner_loss) * grad_x(meta_loss)
                grad_x_meta_loss = torch.autograd.grad(meta_loss(x_hat, x, self.hparams), x_hat)[0]
                
                self.c.requires_grad_()
                if self.hparams.outer.auto_cond_log:
                    cond_log_grad = torch.autograd.grad(log_cond_likelihood_loss\
                        (self.c, y, self.A, x_hat, self.hparams, self.loss_scale, self.efficient_inp), x_hat, create_graph=True)[0]
                else:
                    cond_log_grad = gradient_log_cond_likelihood(self.c, y, self.A, x_hat, self.hparams, self.loss_scale)  

                meta_grad -= hessian_vector_product(self.c, cond_log_grad, grad_x_meta_loss, self.hparams)

            else: 
                raise NotImplementedError
            
            x_hat.requires_grad_(False)
            self.c.requires_grad_(False)

            num_batches -= 1

        #(3) Update the hyperparam weights
        self.c.requires_grad()
        if type(self.c.grad) == type(None): #dummy update to make sure grad is initialized
            dummy_loss = torch.sum(self.c)
            dummy_loss.backward()
        
        meta_grad /= n_samples
        self.c.grad.copy_(meta_grad)
        self.meta_opt.step()

        self.c.requires_grad_(False)

        cur_meta_loss /= n_samples

        return meta_grad.detach().cpu(), cur_meta_loss

    def val_or_test(self, validate):
        if validate:
            val_test = 'VALIDATION'
            cur_loader = self.val_loader
        else:
            val_test = 'TESTING'
            cur_loader = self.test_loader

        if self.hparams.outer.verbose:
            print("\nBEGINNING " + val_test + "\n")
            start = time()

        cur_loss = 0.0
        n_samples = 0

        for i, (x, _) in enumerate(cur_loader):
            n_samples += x.shape[0]

            x = x.to(self.hparams.device)
            y = get_measurements(self.A, x, self.hparams, self.efficient_inp)

            x_mod = torch.rand(x.shape, device=self.hparams.device)
            x_hat = SGLD_inverse(self.c, y, self.A, x_mod, self.model, self.sigmas, self.hparams)

            cur_loss += meta_loss(x_hat, x, self.hparams).item()
        
        cur_loss /= n_samples

        if validate:
            self.val_losses.append(cur_loss)

            if len(self.val_losses) > 1 and self.val_losses[self.best_val_iter] > cur_loss:
                self.best_iter = self.global_iter
                self.best_val_iter = len(self.val_losses)-1
                if self.hparams.outer.verbose:
                    print("\nNEW BEST VAL LOSS: ", cur_loss, "\n")
            elif (len(self.val_losses) - self.best_val_iter) >= 3 and self.hparams.outer.lr_decay:
                self.meta_scheduler.step()
                if self.hparams.outer.verbose :
                    print("\nVAL LOSS HASN'T IMPROVED IN 2 ITERS; DECAYING LR\n")
        else:
            self.test_losses.append(cur_loss)
            
        if self.hparams.outer.verbose:
            end = time()
            print("\n" + val_test + " LOSS: ", cur_loss)
            print(val_test + " TIME: ", str(end - start), " S\n")

        return
