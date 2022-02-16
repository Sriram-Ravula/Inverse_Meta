import numpy as np
import torch

def l2_loss(x_hat, x_true, reduce_dims=None):
    if reduce_dims is None:
        reduce_dims = (0, 1, 2, 3)
    
    return 0.5 * torch.sum((x_hat - x_true)**2, reduce_dims)

def l1_loss(c, scale=1):
    return scale * torch.norm(c, p=1)

def meta_loss(x_hat, x_true, reduce_dims=None, c=None, measurement_loss=False, \
    meta_loss_type='l2', reg_hyperparam=False, reg_hyperparam_type='l1', reg_hyperparam_scale=1,
    ROI_loss=False, ROI=None):

    if measurement_loss:
        raise NotImplementedError("Meta measurement loss not supported!")
    if ROI_loss or ROI is not None:
        raise NotImplementedError("Meta ROI loss not supported!")
    
    if meta_loss_type == 'l2':
        loss_1 = l2_loss(x_hat, x_true, reduce_dims)
    else:
        raise NotImplementedError("Meta loss type not supported")
    
    if reg_hyperparam:
        if reg_hyperparam_type == 'l1':
            loss_2 = l1_loss(c, reg_hyperparam_scale)
        else:
            raise NotImplementedError("Meta regularization type not supported!")
    else:
        loss_2 = 0
    
    return [loss_1, loss_2, loss_1+loss_2] #1st term is fidelity, 2nd is regularization, last is total

def get_meta_grad(x_hat, x_true, reduce_dims=None, c=None, measurement_loss=False, \
    meta_loss_type='l2', reg_hyperparam=False, reg_hyperparam_type='l1', reg_hyperparam_scale=1,
    ROI_loss=False, ROI=None, use_autograd=True, retain_graph=False, create_graph=False):
    
    if not use_autograd:
        raise NotImplementedError("Explicit meta loss not implemented!")
    
    grad_flag_x = x_hat.requires_grad
    x_hat.requires_grad_()
    meta_grad_x = torch.autograd.grad(meta_loss(x_hat, x_true, reduce_dims, c, measurement_loss, meta_loss_type,\
                                             reg_hyperparam, reg_hyperparam_type, reg_hyperparam_scale, ROI_loss, ROI)[-1], 
                                    x_hat, retain_graph=retain_graph, create_graph=create_graph)[0]
    x_hat.requires_grad_(grad_flag_x)

    if reg_hyperparam:
        grad_flag_c = c.requires_grad
        c.requires_grad_()
        meta_grad_c = torch.autograd.grad(meta_loss(x_hat, x_true, reduce_dims, c, measurement_loss, meta_loss_type,\
                                                reg_hyperparam, reg_hyperparam_type, reg_hyperparam_scale, ROI_loss, ROI)[-1], 
                                        c, retain_graph=retain_graph, create_graph=create_graph)[0]
        c.requires_grad_(grad_flag_c)
    else:
        meta_grad_c = 0

    return [meta_grad_x, meta_grad_c]
