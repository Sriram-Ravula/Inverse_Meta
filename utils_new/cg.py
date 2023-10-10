"""
Implements conjugate gradient for measurement consistency as Pytorch modules.

Based on functions from https://github.com/utcsilab/LOUPE_MoDL/
"""

import torch
from problems.fourier_multicoil import MulticoilForwardMRINoMask

class ZConjGrad(torch.nn.Module):
    """
    A class which implements conjugate gradient descent as a torch module.
    This implementation of conjugate gradient descent works as a standard torch module, with the functions forward
        and get_metadata overridden. 
        
    Solves \argmin_x ||A(x) - b||^2 + \lambda ||x - x_init||^2
    - closed-form solution is: x = (A^* A + \lambda I)^-1 (A^*(b) + \lambda x_init)
        - A^* A is equivalent to conjugate_forward(forward( ))
    - actually solves: (A^* A + \lambda I)x = A^*(b) + \lambda x_init
        - solves for x
    
    Args:
        rhs (Tensor): The residual vector b in some conjugate gradient descent algorithms.
            - (A^*(b) + \lambda x_init) 
        Aop_fun (func): A function performing the A matrix operation.
            - A^* A
            - must be a callable
            - for multi-coil MRI, involves point-wise multiplication and sum over coils
        max_iter (int): Maximum number of times to run conjugate gradient descent.
        l2lam (float): The L2 lambda, or regularization parameter (must be positive).
            - \lambda
        eps (float): Determines how small the residuals must be before termination.
        verbose (bool): If true, prints extra information to the console.
    
    Attributes:
        rhs (Tensor): The residual vector, b in some conjugate gradient descent algorithms.
        Aop_fun (func): A function performing the A matrix operation.
        max_iter (int): The maximum number of times to run conjugate gradient descent.
        l2lam (float): The L2 lambda regularization parameter.
        eps (float): Minimum residuals for termination.
        verbose (bool): Whether or not to print extra info to the console.
    """

    def __init__(self, rhs, Aop_fun, max_iter=20, l2lam=0., eps=1e-6, verbose=True):
        super(ZConjGrad, self).__init__()

        self.rhs = rhs
        self.Aop_fun = Aop_fun
        self.max_iter = max_iter
        self.l2lam = l2lam
        self.eps = eps
        self.verbose = verbose

        self.num_cg = None

    def forward(self, x):
        """Performs one forward pass through the conjugate gradient descent algorithm.
        Args:
            x (Tensor): The input to the gradient algorithm.
        Returns:
            The forward pass on x.
        """
        x, num_cg = zconjgrad(x, self.rhs, self.Aop_fun, max_iter=self.max_iter, l2lam=self.l2lam, eps=self.eps, verbose=self.verbose)
        self.num_cg = num_cg
        return x

    def get_metadata(self):
        """Accesses metadata for the algorithm.
        Returns:
            A dict containing metadata.
        """

        return {
                'num_cg': self.num_cg,
                }

def get_Aop_fun(mask, s_maps):
    """
    Helper function that returns a callable A^*A operation given coil maps and mask.
    
    Args:
        mask: Sampling pattern - [N, 1, H, W] real-valued torch tensor
        s_maps: Coil sensitiviy maps - [N, C, H, W] complex complex-valued tensor
        
    Returns:
        Aop_fun (func): A function performing the normal equations, A.H * A
            - In: [N, H, W] complex, Out: [N, H, W] complex
    """
    
    def Aop_fun(x):
        FS = MulticoilForwardMRINoMask(s_maps) #In: [N, H, W] complex, Out: [N, C, H, W] complex
        
        Ax = mask * FS(x) #[N, C, H, W] complex, undersampled measurements
        
        return torch.sum(FS.ifft(Ax) * torch.conj(s_maps), axis=1) #[N, H, W] complex

    return Aop_fun

def get_cg_rhs(mask, s_maps, FSx, l2lam, x_init):
    """
    Helper function that prepares and returns the rhs of the inverse problem linear system
        aka (A^*(b) + \lambda x_init).
    Here b = PFSx and x_init is the (un-normalised) initialisation for CG. 

    Args:
        mask (real-valued tensor): Sampling pattern - [N, 1, H, W] real-valued torch tensor.
        s_maps (complex-valued tensor): Coil sensitiviy maps - [N, C, H, W] complex-valued tensor.
        FSx (complex-valued tensor): Fully-sampled ground truth K-space - [N, C, H, W] complex tensor.
        l2lam (float): The L2 lambda, or regularization parameter (must be positive).
        x_init (complex-valued Tensor): The initial input to the algorithm. [N, H, W] complex.
    
    Returns:
        rhs (complex-valued tensor): (A^*(b) + \lambda x_init). [N, H, W] complex.
    """
    
    FS = MulticoilForwardMRINoMask(s_maps) #NOTE dummy instance, just need ifft function
    
    b = mask * FSx #PFSx [N, C, H, W] complex, undersampled measurements
    
    A_conj_b = torch.sum(FS.ifft(b) * torch.conj(s_maps), axis=1) #[N, H, W] complex
    
    return A_conj_b + l2lam * x_init

def zconjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True):
    """
    Conjugate Gradient Algorithm for a complex vector space applied to batches; assumes the first index is batch size.
    
    Args:
        x (complex-valued Tensor): The initial input to the algorithm.
            - [N, H, W] complex (could convert from [N, 2, H, W] real)
        b (complex-valued Tensor): The residual vector
            - [N, H, W] complex 
        Aop_fun (func): A function performing the normal equations, A.H * A
            - In: [N, H, W] complex, Out: [N, H, W] complex
        max_iter (int): Maximum number of times to run conjugate gradient descent.
        l2lam (float): The L2 lambda, or regularization parameter (must be positive).
        eps (float): Determines how small the residuals must be before termination…
        verbose (bool): If true, prints extra information to the console.
    
    Returns:
    	A tuple containing the output vector x and the number of iterations performed.
    """

    # the first calc of the residual may not be necessary in some cases...
    r = b - (Aop_fun(x) + l2lam * x)
    p = r

    rsnot = zdot_single_batch(r).real
    rsold = rsnot
    rsnew = rsnot

    eps_squared = eps ** 2

    reshape = (-1,) + (1,) * (len(x.shape) - 1)

    num_iter = 0

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=itemize(torch.sqrt(rsnew))))

        if rsnew.max() < eps_squared:
            break

        Ap   = Aop_fun(p) + l2lam * p
        pAp  = zdot_batch(p, Ap).real # !!! Force cast to real
        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = zdot_single_batch(r).real

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r
        num_iter += 1

    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x, num_iter

def itemize(x):
    """
    Converts a Tensor into a list of Python numbers.
    """
    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()
    
def zdot_batch(x1, x2):
    """
    Complex dot product of two complex-valued multidimensional Tensors
    """
    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)

def zdot_single_batch(x):
    """
    Same, applied to self --> squared L2-norm
    """
    return zdot_batch(x, x)
