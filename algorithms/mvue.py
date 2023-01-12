import numpy as np
import torch
from datasets.mri_dataloaders import get_mvue

class Dummy:
    def __init__(self):
        self.s_maps = None

class MVUE_solution:
    def __init__(self, hparams, args, c, device=None):
        self.hparams = hparams
        self.args = args
        self.device = device

        self.c = c.clone().cpu().numpy()
        self.H_funcs = Dummy()
    
    def __call__(self, x_mod, y):
        #out [N, Coils, H, W]
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])

        #convert to numpy
        y = y.clone().cpu().numpy() 
        maps = self.H_funcs.s_maps.clone().cpu().numpy()

        #make the proper measurements
        y = self.c[None, None, :, :] * y

        estimated_mvue = torch.tensor(get_mvue(y, maps), device=x_mod.device) #[N, H, W] complex
        estimated_mvue = torch.view_as_real(estimated_mvue) #[N, H, W, 2] float
        estimated_mvue = torch.permute(estimated_mvue, (0, 3, 1, 2)) #[N, 2, H, W] float
        estimated_mvue = estimated_mvue.type_as(x_mod).contiguous() #call to contiguous stores tensor in one memory location 

        return estimated_mvue
    
    def set_c(self, c):
        self.c = c.clone().cpu().numpy()
