import torch
import torch.nn.functional as F
import numpy as np
from problems.problem import ForwardOperator

class IdentityOperator(ForwardOperator):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def _make_A(self):
        A_linear = torch.eye(self.hparams.data.image_size)
        A_mask = None
        A_functional = torch.nn.Identity()

        A_dict = {'linear': A_linear,
                  'mask': A_mask,
                  'functional': A_functional}
        
        return A_dict
    
    def forward(self, x, targets=False):
        Ax = x.flatten(start_dim=1) #[N, m]

        if targets:
            Ax = self.add_noise(Ax)
        
        return Ax
    
    def adjoint(self, vec):
        out_c, out_h, out_w = self.hparams.image_shape

        return vec.view(-1, out_c, out_h, out_w)
    
    @torch.no_grad()
    def get_measurements_image(self, x, targets=False):
        orig_shape = x.shape

        return self.forward(x, targets).view(orig_shape)
