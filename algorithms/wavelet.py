from nis import maps
import numpy as np
import torch
import sigpy as sp
import sigpy.mri.app

class Dummy:
    def __init__(self):
        self.s_maps = None

class L1_wavelet:
    def __init__(self, hparams, args, c, device=None):
        self.hparams = hparams
        self.args = args
        self.device = device

        self.reg = self.hparams.net.reg_param

        self.c = c.clone().cpu().numpy().astype(complex)  #[H, W] float
        self.H_funcs = Dummy()

    def __call__(self, x_mod, y):
        #out [1, Coils, H, W]
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])

        #get rid of batch dimension
        y = y.squeeze().clone().cpu().numpy() 
        maps = self.H_funcs.s_maps.squeeze().clone().cpu().numpy()

        l1_solver = sigpy.mri.app.L1WaveletRecon(y=y,
                                                 mps=maps,
                                                 lamda=self.reg,
                                                 weights=self.c)
        x_hat = l1_solver.run() #[H, W] complex

        x_hat = torch.tensor(x_hat) #[H, W] complex tensor
        x_hat = torch.view_as_real(x_hat) #[H, W, 2] float tensor
        x_hat = torch.permute(x_hat, (2, 0, 1)) #[2, H, W] float tensor
        x_hat = x_hat.unsqueeze(0) #[1, 2, H, W] float tensor

        return x_hat.to(x_mod.device)

    def set_c(self, c):
        self.c = c.clone().cpu().numpy().astype(complex) 
        