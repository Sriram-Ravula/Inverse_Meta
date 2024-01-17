import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sigpy.mri

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
        self.acs_idx = np.arange((n - self.num_acs_lines) // 2, (n + self.num_acs_lines) // 2)
        self.always_on_idx = self._init_always_on_idx()

        #(2) set the number of learnable parameters and adjust sparsity for acs
        if self.hparams.mask.sample_pattern in ['horizontal', 'vertical']:
            #make sure that always on index doesn't clash with acs region
            self.always_on_idx = np.setdiff1d(self.always_on_idx, self.acs_idx)
            
            #location in an n-sized array to insert our m-sized parameters
            self.insert_mask_idx = np.array([i for i in range(n) if i not in self.acs_idx and i not in self.always_on_idx])

            self.m = n - self.num_acs_lines - len(self.always_on_idx)

            self.sparsity_level = (n/R - self.num_acs_lines - len(self.always_on_idx)) / self.m

        elif self.hparams.mask.sample_pattern == '3D':
            flat_n_inds = np.arange(n**2).reshape(n,n)
            self.acs_idx = flat_n_inds[self.acs_idx[:, None], self.acs_idx].flatten() #fancy indexing grabs a square from center

            #make sure that always on index doesn't clash with acs region
            self.always_on_idx = np.setdiff1d(self.always_on_idx, self.acs_idx)
            
            self.insert_mask_idx = np.array([i for i in range(n**2) if i not in self.acs_idx and i not in self.always_on_idx])

            self.m = n**2 - self.num_acs_lines**2 - len(self.always_on_idx)

            self.sparsity_level = ((n**2)/R - self.num_acs_lines**2 - len(self.always_on_idx)) / self.m
        
        elif self.hparams.mask.sample_pattern == '3D_circle':
            #Make the grids to be used for radius estimation
            x = y = (torch.arange(n) - (n-1)/2) / ((n-1)/2)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=0)
            square_radius_grid = torch.sum(torch.square(grid), dim=0)

            flat_n_inds = np.arange(n**2).reshape(n,n)
            self.corner_idx = flat_n_inds[square_radius_grid > 1].flatten()
            self.acs_idx = flat_n_inds[self.acs_idx[:, None], self.acs_idx].flatten() #fancy indexing grabs a square from center

            #make sure that always on index doesn't clash with acs region or corners
            self.always_on_idx = np.setdiff1d(self.always_on_idx, self.acs_idx)
            self.always_on_idx = np.setdiff1d(self.always_on_idx, self.corner_idx)

            self.insert_mask_idx = np.array([i for i in range(n**2) if i not in self.acs_idx and i not in self.corner_idx and i not in self.always_on_idx])

            self.m = n**2 - self.num_acs_lines**2 - self.corner_idx.size - len(self.always_on_idx)

            self.sparsity_level = ((n**2)/R - self.num_acs_lines**2 - len(self.always_on_idx)) / self.m

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
                probs = torch.ones(self.m, device=self.device) * 0.5
            elif init_method == "random":
                probs = torch.rand(self.m, device=self.device)
            elif init_method == "gaussian_psf":
                if '3D' in self.hparams.mask.sample_pattern:
                    x = y = (torch.arange(n, device=self.device) - (n-1)/2) / ((n-1)/2)
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
                    grid = torch.stack([grid_x, grid_y], dim=0)
                    radius_grid = torch.sum(torch.square(grid), dim=0)
                else:
                    grid = (torch.arange(n, device=self.device) - (n-1)/2) / ((n-1)/2)
                    radius_grid = torch.square(grid)
                std = 0.3
                normalizing_constant = 1 / (2 * np.pi * std**2)
                gauss_psf =  normalizing_constant * torch.exp(-radius_grid / (2 * std**2))
                probs = gauss_psf.flatten()[self.insert_mask_idx]
                    
            self.weights = torch.special.logit(probs, eps=1e-3)
            
        self.normalize_probs()
        self.weights.requires_grad_()

        return

    def _init_always_on_idx(self):
        """
        Initialises the set of k-space points to always sample.
        
        Returns always_on_idx: numpy array of int64
        """
        seed_init = getattr(self.hparams.mask, 'seed_init', None)
        
        if seed_init is None:
            always_on_idx = np.empty(0, dtype=np.int64)
        elif "poisson" in seed_init:
            R = int(seed_init.split("-")[-1])
            n = self.hparams.data.image_size
            
            seed_mask = sigpy.mri.poisson(img_shape=(n, n), accel=R, seed=self.hparams.seed - 1, 
                                          crop_corner=True, dtype=float)
            always_on_idx = np.flatnonzero(seed_mask)
        
        return always_on_idx
    
    @torch.no_grad()
    def normalize_probs(self):
        """
        Projects current weights/logits onto the set with correct mean probability.
        
        To be used for projected gradient descent. 
        """
        probs = torch.sigmoid(self.weights)
        
        mu = torch.mean(probs)
        if mu >= self.sparsity_level:
            normed_probs = (self.sparsity_level / mu) * probs
        else:
            normed_probs = 1 - (1 - self.sparsity_level)/(1 - mu) * (1 - probs)
        
        projected_logits = torch.special.logit(normed_probs, eps=1e-3)
        
        self.weights.copy_(projected_logits)
    
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

    def _get_furthest_point(self, proposed_point_inds):
        """
        Given a list of proposed point indexes to add to the current active list, 
            select and return the point that is furthest from its nearest current 
            active point.
        """
        if len(self.always_on_idx) < 1:
            return 0
        
        n = self.hparams.data.image_size
        
        x = y = (torch.arange(n, device=self.device) - (n-1)/2) / ((n-1)/2)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        xy_grid = torch.stack([grid_x, grid_y], dim=0)
        xy_grid = torch.flatten(xy_grid, start_dim=1) #[2, n^2] grid of (x, y) points
        
        proposed_xy = xy_grid[:, proposed_point_inds] #[2, |proposed_point_inds|] array of (x, y) coords of proposed points
        active_xy = xy_grid[:, self.always_on_idx] #[2, |active_inds|] array of (x, y) coords of active points
        
        diff_xy = proposed_xy[..., None] - active_xy[:, None, ...] #[2, |proposed_point_inds|, |active_inds|] 2D differences
        squared_dists = torch.sum(torch.square(diff_xy), dim=0) #[|proposed_point_inds|, |active_inds|]  squared distances
        
        row_mins = torch.min(squared_dists, dim=1)[0] #[|proposed_point_inds|] array of distance to closest point for each proposed point
        out_idx = torch.argmax(row_mins)
        
        return out_idx

    def max_min_dist_step(self, k=1):
        n = self.hparams.data.image_size
        R = self.hparams.mask.R
        
        top_k_inds = torch.topk(-self.weights.grad, k=k)[1].tolist()
        selected_point_idx = self._get_furthest_point(self.insert_mask_idx[top_k_inds])
        top_k_inds = [top_k_inds[selected_point_idx]]
        
        self.always_on_idx = np.append(self.always_on_idx, self.insert_mask_idx[top_k_inds])
        
        self.insert_mask_idx = np.delete(self.insert_mask_idx, top_k_inds)
        self.m = self.m - len(top_k_inds)
        if '3D' in self.hparams.mask.sample_pattern:
            self.sparsity_level = ((n**2)/R - self.num_acs_lines**2 - len(self.always_on_idx)) / self.m
        else:
            self.sparsity_level = (n/R - self.num_acs_lines - len(self.always_on_idx)) / self.m
        
        keep_inds = [i for i in range(self.weights.numel()) if i not in top_k_inds]
        weights = torch.empty(self.m,
                              dtype=self.weights.dtype,
                              layout=self.weights.layout,
                              device=self.weights.device,
                              requires_grad=True)
        weights.data = self.weights[keep_inds].data
        self.weights = weights
        
        return