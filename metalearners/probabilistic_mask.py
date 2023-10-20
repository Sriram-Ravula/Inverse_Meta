import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Probabilistic_Mask:
    def __init__(self, hparams, device, num_acs_lines=20):
        super().__init__()

        #Expects all info to be in hparams.mask
        #hparams.mask.sample_pattern in ["horizontal", "vertical", "3D"]
        #hparams.mask.R - NOTE this will be the true R with center sampled by default!
        self.hparams = hparams
        self.device = device

        self.num_acs_lines = num_acs_lines #number of lines to keep for 1D, side length ratio of central square for 3D

        self._init_mask()

    def _init_mask(self):
        """
        Initialises locations of acs lines, sparsity level, and the learnable mask of logits
        """
        n = self.hparams.data.image_size
        R = self.hparams.mask.R

        #(1) set the number and location of acs lines
        acs_idx = np.arange((n - self.num_acs_lines) // 2, (n + self.num_acs_lines) // 2)

        #(2) set the number of learnable parameters and adjust sparsity for acs
        if self.hparams.mask.sample_pattern in ['horizontal', 'vertical']:
            #location in an n-sized array to insert our m-sized parameters
            self.insert_mask_idx = np.array([i for i in range(n) if i not in acs_idx])

            self.m = n - self.num_acs_lines

            self.sparsity_level = (n/R - self.num_acs_lines) / self.m

        elif self.hparams.mask.sample_pattern == '3D':
            flat_n_inds = np.arange(n**2).reshape(n,n)
            acs_idx = flat_n_inds[acs_idx[:, None], acs_idx].flatten() #fancy indexing grabs a square from center
            self.insert_mask_idx = np.array([i for i in range(n**2) if i not in acs_idx])

            self.m = n**2 - self.num_acs_lines**2

            self.sparsity_level = ((n**2)/R - self.num_acs_lines**2) / self.m
        
        elif self.hparams.mask.sample_pattern == '3D_circle':
            #Make the grids to be used for radius estimation
            x = y = (torch.arange(n) - (n-1)/2) / ((n-1)/2)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=0)
            square_radius_grid = torch.sum(torch.square(grid), dim=0)
            
            flat_n_inds = np.arange(n**2).reshape(n,n)
            self.corner_idx = flat_n_inds[square_radius_grid > 1].flatten()
            acs_idx = flat_n_inds[acs_idx[:, None], acs_idx].flatten() #fancy indexing grabs a square from center
            self.insert_mask_idx = np.array([i for i in range(n**2) if i not in acs_idx and i not in self.corner_idx])

            self.m = n**2 - self.num_acs_lines**2 - self.corner_idx.size

            self.sparsity_level = ((n**2)/R - self.num_acs_lines**2) / self.m

        else:
            raise NotImplementedError("Fourier sampling pattern not supported!")

        #(3) initialize the weights - logits of a bernouli distribution
        #Pick a distribution we like for the probabilistic mask, then 
        #   take the logits of the entries
        init_method = getattr(self.hparams.mask, 'mask_init', "random")
        
        if init_method == "normal":
            self.weights = torch.randn(self.m, device=self.device)
        else:
            if init_method == "uniform":
                probs = torch.ones(self.m) * 0.5
            elif init_method == "random":
                probs = torch.rand(self.m)
            self.weights = torch.special.logit(probs, eps=1e-3).to(self.device)
            
        self.weights.requires_grad_()

        return
    
    def _normalize_probs(self, prob_mask):
        """
        Given a mask of probabilities, renormalizes it to have a desired mean value.

        Note - this requires probability inputs, use sigmoid on input first if giving logits
        """
        mu = torch.mean(prob_mask)

        if mu >= self.sparsity_level:
            return (self.sparsity_level / mu) * prob_mask
        else:
            return 1 - (1 - self.sparsity_level)/(1 - mu) * (1 - prob_mask)
    
    def _sample_mask(self, prob_mask, tau=0.5):
        """
        Given a mask of probabilities, samples a realization where each entry is a bernouli variable.

        Uses the Gumbel straight-through estimator 
        """
        #Sampling requires us to draw a gumbel sample for each category/binary outcome
        prob_mask_01 = torch.stack((1. - prob_mask, prob_mask), dim=1) #[m, 2]

        #pytorch function requires un-normalized log-probabilities
        gumbel_mask_sample = F.gumbel_softmax(torch.log(prob_mask_01), tau=tau, hard=True)[:, 1] #[m]

        return gumbel_mask_sample
    
    def _reshape_mask(self, raw_mask):
        """
        Given a flat, raw mask, re-shapes it properly and applies ACS lines
        """
        n = self.hparams.data.image_size
        sample_pattern = self.hparams.mask.sample_pattern

        #start with all ones for acs, then apply our raw mask around acs
        flat_mask = torch.ones(n**2 if '3D' in sample_pattern else n, 
                                device=raw_mask.device, dtype=raw_mask.dtype)
        flat_mask[self.insert_mask_idx] = raw_mask
        
        if sample_pattern == "3D_circle":
            flat_mask[self.corner_idx] = 0

        if sample_pattern == 'horizontal':
            out_mask = flat_mask.unsqueeze(1).repeat(1, n) 

        elif sample_pattern == 'vertical':
            out_mask = flat_mask.unsqueeze(0).repeat(n, 1)

        elif '3D' in sample_pattern:
            out_mask = flat_mask.view(n, n)

        return out_mask
    
    def sample_mask(self, tau=0.5):
        """
        Samples a binary mask based on our learned logit values
        """
        probs = torch.sigmoid(self.weights)
        normed_probs = self._normalize_probs(probs)
        flat_sample = self._sample_mask(normed_probs, tau=tau)
        sampled_mask = self._reshape_mask(flat_sample)
        
        return sampled_mask

    @torch.no_grad()
    def get_prob_mask(self):
        """
        Returns a mask of re-normalized probabilities based on learned logits.
        """
        probs = torch.sigmoid(self.weights)
        normed_probs = self._normalize_probs(probs)
        prob_mask = self._reshape_mask(normed_probs)

        return prob_mask
    
    @torch.no_grad()
    def get_max_mask(self):
        """
        Returns a binary mask with acceleration R by keeping only the top logits.
        """
        k = int(self.m * (1 - self.sparsity_level))
        smallest_kept_val = torch.kthvalue(self.weights, k)[0]
        under_idx = self.weights < smallest_kept_val

        weights_copy = torch.ones_like(self.weights)
        weights_copy[under_idx] = 0.

        max_mask = self._reshape_mask(weights_copy)

        return max_mask
