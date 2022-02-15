import torch
import numpy as np
from problems.problem import ForwardOperator

class InpaintingOperator(ForwardOperator):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def forward(self, x, targets=False):
        if self.hparams.problem.efficient_inp:
            Ax = self.A_mask * x #[N, C, H, W]
            Ax = Ax.flatten(start_dim=1)[:, self.kept_inds] #[N, m]
        else:
            Ax = torch.mm(self.A_linear, torch.flatten(x, start_dim=1).T).T #[N, m]
        
        if targets:
            Ax = self.add_noise(Ax)
        
        return Ax

    def adjoint(self, vec):
        out_c, out_h, out_w = self.hparams.image_shape
        ans = torch.mm(self.A_linear.T, vec.T).T #[N, n]

        return ans.view(-1, out_c, out_h, out_w)

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

    @torch.no_grad()
    def get_measurements_image(self, x, targets=False):
        orig_shape = x.shape

        Ax = self.A_mask * x #[N, C, H, W]
        Ax = Ax.flatten(start_dim=1) #[N, n]
        
        if targets:
            Ax[:, self.kept_inds] = self.add_noise(Ax[:, self.kept_inds]) #only apply noise to relevant [N, m]
        
        Ax = Ax.view(orig_shape) #[N, C, H, W]

        return Ax
    