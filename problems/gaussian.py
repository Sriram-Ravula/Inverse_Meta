import torch
import numpy as np
from problems.problem import ForwardOperator

class GaussianOperator(ForwardOperator):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def _make_A(self):
        m = self.hparams.problem.num_measurements
        n = self.hparams.data.n_input

        A_linear = (1 / np.sqrt(m)) * torch.randn(m, n)
        A_mask = None
        A_functional = None

        A_dict = {'linear': A_linear,
                  'mask': A_mask,
                  'functional': A_functional}
        
        return A_dict

    def forward(self, x, targets=False):
        Ax = torch.mm(self.A_linear, torch.flatten(x, start_dim=1).T).T #[N, m]

        if targets:
            Ax = self.add_noise(Ax)
        
        return Ax

    def adjoint(self, vec):
        out_c, out_h, out_w = self.hparams.image_shape
        ans = torch.mm(self.A_linear.T, vec.T).T #[N, n] 

        return ans.view(-1, out_c, out_h, out_w) #[N, C, H, W]
    
    @torch.no_grad()
    def get_measurements_image(self, x, targets=False):
        return None
