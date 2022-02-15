import torch
import torch.nn.functional as F
import numpy as np
from problems.problem import ForwardOperator

class SuperresOperator(ForwardOperator):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def _make_A(self):
        def downsample(x):
            return F.avg_pool2d(x, self.hparams.problem.downsample_factor)

        A_linear = None
        A_mask = None
        A_functional = downsample

        A_dict = {'linear': A_linear,
                  'mask': A_mask,
                  'functional': A_functional}
        
        return A_dict
    
    def forward(self, x, targets=False):
        Ax = F.avg_pool2d(x, self.hparams.problem.downsample_factor) #[N, C, H//downsample, W//downsample]
        Ax = Ax.flatten(start_dim=1) #[N, m]

        if targets:
            Ax = self.add_noise(Ax)
        
        return Ax
    
    def adjoint(self, vec):
        C = self.hparams.data.num_channels
        down_size = self.hparams.data.image_size // self.hparams.problem.downsample_factor

        vec = vec.view(-1, C, down_size, down_size)

        return F.interpolate(vec, scale_factor=self.hparams.problem.downsample_factor)
    
    @torch.no_grad()
    def get_measurements_image(self, x, targets=False):
        Ax = self.forward(x, targets) #[N, m]

        return self.adjoint(Ax) #[N, C, H, W]
