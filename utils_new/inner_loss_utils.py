import numpy as np
import torch

def log_cond_likelihood_loss(c, y, A, x,
                             scale=1.,
                             reduce_dims=None):

    #if not given the dimensions to reduce, just reduce them all
    if reduce_dims is None:
        reduce_dims = tuple(np.arange(y.dim()))

    #take the measurements
    Ax = A(x)
    resid = Ax - y

    #match broadcast dimensions
    if resid.shape[-1] != c.shape[-1]:
        c = c.unsqueeze(-1)

    #(1/2) ||Diag(sqrt(c))(Ax-y)||^2
    loss = scale * 0.5 * torch.sum(c * (resid ** 2), reduce_dims) 

    return loss

def get_likelihood_grad(c, y, A, x,
                        scale=1.,
                        reduce_dims=None,
                        retain_graph=False,
                        create_graph=False):
    grad_flag_x = x.requires_grad
    x.requires_grad_()
    likelihood_grad = torch.autograd.grad(log_cond_likelihood_loss(c, y, A, x, scale, reduce_dims),
                                        x, retain_graph=retain_graph, create_graph=create_graph)[0]
    x.requires_grad_(grad_flag_x)

    return likelihood_grad
