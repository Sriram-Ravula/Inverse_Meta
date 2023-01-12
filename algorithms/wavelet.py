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

        self.c = c.clone().cpu().numpy()#.astype(complex)  #[H, W] float
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

        #make a container for the final solutions
        x_out = torch.zeros_like(x_mod)

        for i in range(y.shape[0]):
            y_i = y[i]
            maps_i = maps[i]

            l1_solver = sigpy.mri.app.L1WaveletRecon(y=y_i,
                                                    mps=maps_i,
                                                    lamda=self.reg,
                                                    solver="GradientMethod" if self.reg > 0.0 else "ConjugateGradient")
            x_hat = l1_solver.run() #[H, W] complex

            x_hat = torch.tensor(x_hat) #[H, W] complex tensor
            x_hat = torch.view_as_real(x_hat) #[H, W, 2] float tensor
            x_hat = torch.permute(x_hat, (2, 0, 1)) #[2, H, W] float tensor
            x_out[i] = x_hat.to(x_out.device)

        return x_out

    def set_c(self, c):
        self.c = c.clone().cpu().numpy()#.astype(complex) 
        