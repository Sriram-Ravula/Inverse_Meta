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
        self.register_buffer('A_functional', A_dict['functional'])

        self.register_buffer('kept_inds', self._make_kept_inds())

        self.register_buffer('noisy', self.hparams.problem.add_noise)
        if self.hparams.problem.add_noise and self.hparams.problem.noise_type == 'gaussian_nonwhite':
            self.register_buffer('noise_vars', torch.rand(self.hparams.problem.y_shape))
        else:
            self.register_buffer('noise_vars', None)
    
    def get_A(self):
        return self.A_linear
    
    def get_mask(self):
        return self.A_mask
    
    def get_functional(self):
        return self.A_functional
    
    def get_kept_inds(self):
        return self.kept_inds

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
        Assumes the input x will have shape [C, H, W] 
        """
        if self.A_mask is None:
            return None

        #if the binary mask is only two-dimensional [H, W] and the images are 3-dimensional [C, H, W]
        #we need to properly resize the mask to give the correct indices
        if len(self.A_mask.shape) < self.hparams.data.image_shape: 
            kept_inds = (self.A_mask.unsqueeze(0).repeat(self.hparams.data.num_channels, 1, 1).flatten()>0).nonzero(as_tuple=False).flatten()
        else:
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
            y: y=Ax + noise (make_targets=True), y=Ax (make_targets=False)
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
        Constructs and returns additive noise.

        Args:
            Ax: The measurements to add noise to.
                Torch tensor [self.hparams.problem.y_shape].
        """
        if self.hparams.problem.noise_type == 'gaussian':
            noise = torch.randn(Ax.shape).type_as(Ax) * self.hparams.problem.noise_std
        elif self.hparams.problem.noise_type == 'gaussian_nonwhite': 
            noise = torch.randn(Ax.shape).type_as(Ax) * self.hparams.problem.noise_std * self.noise_vars
        else:
            raise NotImplementedError('unsupported type of additive noise')
        
        return Ax + noise

    def get_measurements_image(self, x, targets=False):
        """
        Makes and returns an image of x under the forward operator.
        If not possible to visualise the measurements for this operator, returns None.
        Should be inherited and overrided by child classes.

        Args:
            x: The image to take measurements from.
               Torch tensor [N, C, H, W].
        """
        raise NotImplementedError('subclasses must override get_measurements_image()!') 
    
class InverseProblem(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.register_buffer('A', self.make_A())
        if self.hparams.problem.measurement_type == 'inpaint':
            self.register_buffer('inpaint_mask', self.make_inpaint_mask())
        elif self.hparams.problem.measurement_type == 'fourier':
            self.register_buffer('fourier_mask', self.make_random_mask())

    def make_A(self):
        """
        Returns a forward operator for making the measurements.
        For Gaussian measurements, returns an [m, n] matrix.
        For superresolution, identity, and Fourier, returns None (the forward operator is a functional).
        For inpainting, if using efficient inpainting, returns None. 
            else, returns an [m, n] subsampled identity matrix.
        """
        A_type = self.hparams.problem.measurement_type

        if A_type == 'gaussian':
            A = (1 / np.sqrt(self.hparams.problem.num_measurements)) * \
                torch.randn(self.hparams.problem.num_measurements, self.hparams.data.n_input)
        elif A_type == 'superres':
            A = None
        elif A_type == 'inpaint':
            if self.hparams.problem.efficient_inp:
                A = None
            else:
                A = self.make_A_inpaint()
        elif A_type == 'identity':
            A = None 
        elif A_type == 'fourier': #TODO TEST THIS 
            A = None
        else:
            raise NotImplementedError

        return A

    def make_inpaint_mask(self):
        """
        Returns a [H, W] binary mask (torch tensor) with square 0 region in center for inpainting.
        """
        image_size = self.hparams.data.image_size
        inpaint_size = self.hparams.problem.inpaint_size

        margin = (image_size - inpaint_size) // 2
        mask = torch.ones(image_size, image_size)
        mask[margin:margin+inpaint_size, margin:margin+inpaint_size] = 0

        return mask
    
    def make_A_inpaint(self):
        """
        Returns an [m, n=C*H*W] forward matrix (torch tensor) for inpainting.   
        """
        mask = self.make_inpaint_mask().unsqueeze(0).repeat(self.hparams.data.num_channels, 1, 1).numpy()
        mask = mask.reshape(1, -1)
        A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
        A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

        return torch.from_numpy(A).float() 
    
    def make_random_mask(self):
        """
        Returns a [H, W] binary mask (torch tensor) with random 0s for Fourier subsampling.
        #NOTE right now the num_measurements are actually 3x num_measurements since we apply same randomness across channels
        """
        image_size = self.hparams.data.image_size

        kept_meas = np.random.choice(image_size**2, size=self.hparams.problem.num_measurements, replace=False)

        mask = torch.zeros(image_size**2)
        mask[kept_meas] = 1
        mask = mask.view(image_size, image_size)

        return mask

    def fft(self, x):
        """
        Performs a centered and orthogonal fft in torch >= 1.7
        """
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x
        
    def ifft(self, x):
        """
        Performs a centered and orthogonal ifft in torch >= 1.7
        """
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    def get_Fourier_meas(self, x):
        """
        Calculates and returns subsampled Fourier measurements.
        """
        fft_x = self.fft(x) #complex tensor [N, C, H, W]
        fft_x = fft_x * self.fourier_mask
        fft_x = torch.view_as_real(fft_x) #real tensor [N, C, H, W, 2]

        return fft_x

    def get_measurements(self, x):
        """
        Return measurements A(x) of the given image.
        """
        A_type = self.hparams.problem.measurement_type

        if A_type == 'gaussian':
            Ax = torch.mm(self.A, torch.flatten(x, start_dim=1).T).T #[N, m]
        elif A_type == 'inpaint':
            if self.hparams.problem.efficient_inp:
                Ax = self.inpaint_mask * x #[N, C, H, W]
            else:
                Ax = torch.mm(self.A, torch.flatten(x, start_dim=1).T).T #[N, m]
        elif A_type == 'superres':
            Ax = F.avg_pool2d(x, self.hparams.problem.downsample_factor) #[N, C, H//downsample_factor, W//downsample_factor]
        elif A_type == 'identity':
            I = torch.nn.Identity()
            Ax = I(x) #[N, C, H, W]
        elif A_type == 'fourier':
            Ax = self.get_Fourier_meas(x) #[N, C, H, W, 2]
        else:
            raise NotImplementedError #TODO implement circulant!!
        
        #Now to add the noise!
