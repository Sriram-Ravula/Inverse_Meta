from utils_new.models import Fixed_Input_UNet
import numpy as np
import torch
import torch.nn.functional as F

class Network_Mask:
    def __init__(self, hparams, device, num_acs_lines=20):
        super().__init__()
        
        self.hparams = hparams
        self.device = device
        self.num_acs_lines = num_acs_lines
    
        self.n = self.hparams.data.image_size
        self.R = self.hparams.mask.R
        
        #(1) Create the sampling pattern masks and variables
        
        self.acs_mask = None #binary mask with 1s in acs region and 0s outside
        self.pattern_mask = None #binary mask with 1s in pattern support and 0s outside 
        self.sparsity_level = None #ratio of (num samples)/(possible sampling spots) in the pattern support
        self.m = None #Total number of supported sampling locations
        
        acs_idx = np.arange((self.n - self.num_acs_lines) // 2, (self.n + self.num_acs_lines) // 2)
        
        if self.hparams.mask.sample_pattern in ['horizontal', 'vertical']:
            raise NotImplementedError("2D Patterns Currently Unsupported!")
        elif self.hparams.mask.sample_pattern == '3D':
            self.acs_mask = torch.zeros((self.n, self.n), requires_grad=False)
            self.acs_mask[acs_idx[:, None], acs_idx] = 1.
            
            self.pattern_mask = torch.ones((self.n, self.n), requires_grad=False)
            
            self.m = self.n**2 - self.num_acs_lines**2
            self.sparsity_level = ((self.n**2)/self.R - self.num_acs_lines**2) / self.m
        elif self.hparams.mask.sample_pattern == '3D_circle':
            self.acs_mask = torch.zeros((self.n, self.n), requires_grad=False)
            self.acs_mask[acs_idx[:, None], acs_idx] = 1.
            
            #Make the grids to be used for radius estimation
            x = y = (torch.arange(self.n) - (self.n-1)/2) / ((self.n-1)/2)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=0)
            square_radius_grid = torch.sum(torch.square(grid), dim=0)
            
            self.pattern_mask = torch.ones((self.n, self.n), requires_grad=False)
            self.pattern_mask[square_radius_grid > 1] = 0.
            
            corner_size = torch.sum(torch.ones((self.n, self.n)) - self.pattern_mask).item()
            self.m = self.n**2 - self.num_acs_lines**2 - corner_size
            self.sparsity_level = ((self.n**2)/self.R - self.num_acs_lines**2) / self.m
        else:
            raise NotImplementedError("Pattern not supported!")
        
        #(2) Make the pattern network
        self.ngf = 8
        self.pattern_net = Fixed_Input_UNet(ngf=self.ngf, output_size=self.n)
        self.pattern_net.train(True) #Net has fixed input; always keep to train() since batchnorm stats dont matter
    
    def parameters(self):
        return self.pattern_net.parameters()
        
    def _normalize_probs(self, prob_mask):
        """
        Given a mask of probabilities, renormalizes it to have a desired mean value.

        Note - this requires probability inputs, use sigmoid on input first if giving logits
        """
        mu = torch.sum(prob_mask) / self.m #NOTE this calcs the mean only in supported areas
        
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
        prob_mask_01 = torch.stack((1. - prob_mask, prob_mask), dim=0) #[2, ...]

        #pytorch function requires un-normalized log-probabilities
        gumbel_mask_sample = F.gumbel_softmax(torch.log(prob_mask_01), tau=tau, hard=True)[1] #[...]

        return gumbel_mask_sample
    
    def sample_mask(self, tau=0.5):
        logits = self.pattern_net().squeeze()
        probs = torch.sigmoid(logits)
        supported_probs = self._mask_unsupported_idxs(probs)
        normed_probs = self._normalize_probs(supported_probs)
        sampled_mask = self._sample_mask(normed_probs, tau=tau)
        return self._apply_masks(sampled_mask)
    
    @torch.no_grad()
    def get_prob_mask(self):
        """
        Returns a mask of re-normalized probabilities based on learned logits.
        """
        logits = self.pattern_net().squeeze()
        probs = torch.sigmoid(logits)
        supported_probs = self._mask_unsupported_idxs(probs)
        normed_probs = self._normalize_probs(supported_probs)
        return self._apply_masks(normed_probs)
    
    @torch.no_grad()
    def get_max_mask(self):
        """
        Returns a binary mask with acceleration R by keeping only the top logits.
        """
        logits = self.pattern_net().squeeze()
        probs = torch.sigmoid(logits)
        supported_probs = self._mask_unsupported_idxs(probs)
        
        k = int(self.m * (1 - self.sparsity_level))
        smallest_kept_val = torch.kthvalue(supported_probs[supported_probs > 0.], k)[0]
        
        max_mask = torch.ones_like(supported_probs)
        max_mask[supported_probs < smallest_kept_val] = 0.
        
        return self._apply_masks(max_mask)

    def _mask_unsupported_idxs(self, input):
        """
        Removes acs region and any other unsupported areas
        """
        return input * (1 - self.acs_mask) * self.pattern_mask

    def _apply_masks(self, input):
        """
        Applies acs region and removes unsupported areas.
        """
        acs_sampled = input * (1 - self.acs_mask) + self.acs_mask
        return acs_sampled * self.pattern_mask
    
