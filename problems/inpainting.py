import torch
import numpy as np

class InpaintingOperator(torch.nn.module):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def forward(self, x, add_noise=False):
        if self.hparams.problem.efficient_inp:
            Ax = self.A_mask * x #[N, C, H, W]
        else:
            Ax = torch.mm(self.A_linear, torch.flatten(x, start_dim=1).T).T #[N, m]
        
        if add_noise:
            Ax = self.add_noise(Ax)
        
        return Ax

    def get_transpose_measurements(self, vec):
        

    def _make_inpaint_mask(self):
        """
        Returns a [H, W] binary mask (torch tensor) with square 0 region in center for inpainting.
        """
        image_size = self.hparams.data.image_size
        inpaint_size = self.hparams.problem.inpaint_size

        margin = (image_size - inpaint_size) // 2
        mask = torch.ones(image_size, image_size)
        mask[margin:margin+inpaint_size, margin:margin+inpaint_size] = 0

        return mask
    
    def _make_A_inpaint(self):
        """
        Returns an [m, n=C*H*W] forward matrix (torch tensor) for inpainting.   
        """
        mask = self.make_inpaint_mask().unsqueeze(0).repeat(self.hparams.data.num_channels, 1, 1).numpy()
        mask = mask.reshape(1, -1)
        A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
        A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

        return torch.from_numpy(A).float() 

    def _make_A(self):
        if self.hparams.problem.efficient_inp:
            A_linear = None
        else:
            A_linear = self._make_A_inpaint()
        A_mask = self._make_inpaint_mask()
        A_functional = None

        A_dict = {'linear': A_linear,
                  'mask': A_mask,
                  'functional': A_functional}
        
        return A_dict

    def get_measurements_image(self, x, add_noise=False):
