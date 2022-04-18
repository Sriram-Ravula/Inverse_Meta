import numpy as np
import torch

def gradient_log_cond_likelihood(c_orig, y, A, x, 
                                 scale=1., 
                                 exp_params=False):
    """
    Explicit gradient for l2 (log conditional likelihood) loss.

    Args:
        c: 
    """
    c_type = len(c_orig.shape)

    if exp_params:
        c = torch.exp(c_orig)
    else:
        c = c_orig
    
    Ax = A(x, targets=False) #don't add noise since we are making a sample
    resid = Ax - y #[N, m]

    if c_type == 0:
        grad = scale * c * A.adjoint(resid) #c * A^T * (Ax - y)
    elif c_type == 1:
        grad = scale * A.adjoint(c * resid) #A^T * Diag(c) * (Ax - y)
    else:
        raise NotImplementedError("Hyperparameter dimensions not supported")
    
    return grad #the adjoint takes care of reshaping properly

def log_cond_likelihood_loss(c_orig, y, A, x, 
                             scale=1., 
                             exp_params=False, 
                             reduce_dims=None,
                             learn_samples=False,
                             sample_pattern=None):

    c_type = len(c_orig.shape)

    if exp_params:
        c = torch.exp(c_orig)
    else:
        c = c_orig
    
    if reduce_dims is None:
        reduce_dims = y.shape
    
    #[N, m] or [N, 2m] (fourier)
    #learn_samples: [N, C, H//D, W//D] (superres), [N, C, H, W] (inpaint), [N, C, H, W, 2] (fourier)
    Ax = A(x, targets=False)
    resid = Ax - y

    #we need to reshape the hyperparameters to be [H, W] (or [H//D, W//D] for superres)
    if learn_samples:
        if sample_pattern == 'random':
            c = c.view(resid.shape[2:4])
        elif sample_pattern == 'horizontal':
            c = c.unsqueeze(1).repeat(1, resid.shape[3])
        elif sample_pattern == 'vertical':
            c = c.unsqueeze(0).repeat(resid.shape[2], 1)
    
    #if there is a trailing 2 (Fourier) then match the dimensions to broadcast
    if resid.shape[-1] != c.shape[-1]:
        c.unsqueeze(-1)

    if c_type == 0:
        loss = scale * c * 0.5 * torch.sum(resid ** 2, reduce_dims) #(c/2) ||Ax - y||^2
    elif c_type == 1:
        loss = scale * 0.5 * torch.sum(c * (resid ** 2), reduce_dims) #(1/2) ||Diag(sqrt(c))(Ax-y)||^2
    else:
        raise NotImplementedError("Hyperparameter dimensions not supported")

    return loss

def get_likelihood_grad(c, y, A, x, use_autograd, 
                        scale=1., 
                        exp_params=False, 
                        reduce_dims=None,
                        learn_samples=False,
                        sample_pattern=None,
                        retain_graph=False, 
                        create_graph=False):
    """
    A method for choosing between gradient_log_cond_likelihood (explicitly-formed gradient)
        and log_cond_likelihood_loss with autograd. 
    """
    if use_autograd:
        grad_flag_x = x.requires_grad
        x.requires_grad_()
        likelihood_grad = torch.autograd.grad(log_cond_likelihood_loss(c, y, A, x, scale, exp_params, reduce_dims, learn_samples, sample_pattern), 
                            x, retain_graph=retain_graph, create_graph=create_graph)[0]
        x.requires_grad_(grad_flag_x)
    else:
        likelihood_grad = gradient_log_cond_likelihood(c, y, A, x, scale, exp_params)
    
    return likelihood_grad
