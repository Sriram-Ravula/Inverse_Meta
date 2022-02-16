import numpy as np
import torch

def gradient_log_cond_likelihood(c, y, A, x, scale=1, exp_params=False):
    """
    Explicit gradient for l2 (log conditional likelihood) loss.

    Args:
        c: 
    """
    c_type = len(c.shape)

    if exp_params:
        c = torch.exp(c)
    
    Ax = A(x) #don't add noise since we are making a sample
    resid = Ax - y #[N, m]

    if c_type == 0:
        grad = scale * c * A.adjoint(resid) #c * A^T * (Ax - y)
    elif c_type == 1:
        grad = scale * A.adjoint(c * resid) #A^T * Diag(c) * (Ax - y)
    elif c_type == 2:
        grad = scale * A.adjoint(torch.mm(torch.mm(c.T, c), resid.T).T) #A^T * C^T * C * (Ax - y)
    else:
        raise NotImplementedError("Hyperparameter dimensions not supported")
    
    return grad #the adjoint takes care of reshaping properly

def log_cond_likelihood_loss(c, y, A, x, scale=1, exp_params=False, reduce_dims=None):
    c_type = len(c.shape)

    if exp_params:
        c = torch.exp(c)
    
    if reduce_dims is None:
        reduce_dims = (0, 1)
    
    Ax = A(x) #don't add noise since we are making a sample
    resid = Ax - y #[N, m]

    if c_type == 0:
        loss = scale * c * 0.5 * torch.sum(resid ** 2, reduce_dims) #(c/2) ||Ax - y||^2
    elif c_type == 1:
        loss = scale * 0.5 * torch.sum(c * (resid ** 2), reduce_dims) #(1/2) ||Diag(sqrt(c))(Ax-y)||^2
    elif c_type == 2:
        interior = torch.mm(c, resid.T).T #[N, m]
        loss = scale * 0.5 * torch.sum(interior ** 2, reduce_dims) #(1/2) ||C(Ax - y)||^2
    else:
        raise NotImplementedError("Hyperparameter dimensions not supported")

    return loss

def get_likelihood_grad(c, y, A, x, use_autograd, scale=1, exp_params=False, reduce_dims=None,\
    retain_graph=False, create_graph=False):
    """
    A method for choosing between gradient_log_cond_likelihood (explicitly-formed gradient)
        and log_cond_likelihood_loss with autograd. 
    """
    if use_autograd:
        grad_flag_x = x.requires_grad
        x.requires_grad_()
        likelihood_grad = torch.autograd.grad(log_cond_likelihood_loss(c, y, A, x, scale, exp_params, reduce_dims), 
                            x, retain_graph=retain_graph, create_graph=create_graph)[0]
        x.requires_grad_(grad_flag_x)
    else:
        likelihood_grad = gradient_log_cond_likelihood(c, y, A, x, scale, exp_params)
    
    return likelihood_grad
