import torch
import torch.fft
import torch.nn.functional as F
import numpy as np

class ForwardOperator(torch.nn.Module):
    def __init__(self, hparams):
        """
        A parent class that defines basic functions of a forward measurement operator.
        Maintains a persistent state.
        Register attributes as buffers for ease of serializaton for multi-GPU, state saving, etc.  
        Should be inherited and overriden by child classes.

        Args:
            hparams: The hyperparameter configuration associated with the inverse problem.
                     Namespace.
        """
        super().__init__()

        self.hparams = hparams

        A_dict = self._make_A()
        self.register_buffer('A_linear', A_dict['linear'])
        self.register_buffer('A_mask', A_dict['mask'])
        self.A_functional = A_dict['functional']

        self.register_buffer('kept_inds', self._make_kept_inds())

        self.noisy = self.hparams.problem.add_noise
        self.register_buffer('noise_vars', self._make_noise())
    
    def get_A(self):
        return self.A_linear
    
    def get_mask(self):
        return self.A_mask
    
    def get_functional(self):
        return self.A_functional
    
    def get_kept_inds(self):
        return self.kept_inds
    
    def _make_noise(self):
        if self.hparams.problem.add_noise and self.hparams.problem.noise_type == 'gaussian_nonwhite':
            noise_vars = torch.rand(self.hparams.problem.num_measurements)
        else:
            noise_vars = None
        
        return noise_vars

    def _make_A(self):
        """
        Constructs and returns the matrix, mask, and functional associated with forward operator A.
        If one of those does not apply to the problem, return None.
        Should be inherited and overrided by child classes.
        """
        raise NotImplementedError('subclasses must override make_A()!') 

    def _make_kept_inds(self):
        """
        Returns the nonzero indices in A_mask, flattened. 
        Assumes the mask is 2D. 
        """
        if self.A_mask is None:
            return None

        kept_inds = (self.A_mask.flatten()>0).nonzero(as_tuple=False).flatten()
        
        return kept_inds
    
    def forward(self, x, targets=False):
        """
        Performs y = A(x) + noise.
        Should be inherited and overrided by child classes.

        Args:
            x: The data to take measurements from.
               Torch tensor [N, C, H, W].
            targets: Whether the forward process is being used to make targets for training/testing.
                     E.G. if True, does y = Ax + noise, False does y = Ax (if noise is applicable). 
                     Bool. 
        Returns:
            y: y=Ax + noise (targets=True), y=Ax (targets=False)
               Torch tensor [N, m].
        """
        raise NotImplementedError('subclasses must override forward()!')

    def adjoint(self, vec):
        """
        Performs A^T (vec), or the equivalent if A is a functional. 
        Should be inherited and overrided by child classes.

        Args:
            vec: The vector to take adjoint measurements from. 
                 Torch tensor [N, m].
        
        Returns:
            y = A^T (vec).
                Torch tensor [N, image_shape]
        """
        raise NotImplementedError('subclasses must override get_adjoint_measurements()!') 
    
    def add_noise(self, Ax):
        """
        Constructs and returns additive noise if the problem calls for it.

        Args:
            Ax: The measurements to add noise to.
                Torch tensor [N, self.hparams.problem.y_shape].
        """
        if self.noisy:
            if self.hparams.problem.noise_type == 'gaussian':
                noise = torch.randn(self.hparams.problem.num_measurements).type_as(Ax) * self.hparams.problem.noise_std
            elif self.hparams.problem.noise_type == 'gaussian_nonwhite': 
                noise = torch.randn(self.hparams.problem.num_measurements).type_as(Ax) * self.hparams.problem.noise_std * self.noise_vars.type_as(Ax)
            else:
                raise NotImplementedError('unsupported type of additive noise')
        
            return Ax + noise.view(Ax.shape[1:])
        else:
            return Ax

    @torch.no_grad()
    def get_measurements_image(self, x, targets=False):
        """
        Makes and returns an image or list of images of x under the forward operator.
        If not possible to visualise the measurements for this operator, returns None.
        Should be inherited and overrided by child classes.

        Args:
            x: The image to take measurements from.
               Torch tensor [N, C, H, W].
        
        Returns:
            y_meas: The measurements in pixel space.
                    Torch tensor [N, C, H, W].
        """
        raise NotImplementedError('subclasses must override get_measurements_image()!') 
    