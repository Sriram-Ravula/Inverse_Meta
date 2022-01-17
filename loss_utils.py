import torch
import torch.nn.functional as F
import numpy as np


def get_A(hparams):
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian':
        A = (1 / np.sqrt(hparams.problem.num_measurements)) * torch.randn(hparams.problem.num_measurements, hparams.data.n_input)
    elif A_type == 'superres':
        A = None #don't explicitly form subsampling matrix
    elif A_type == 'inpaint':
        A = get_A_inpaint(hparams) #TODO make inpainting more efficient!
    elif A_type == 'identity':
        A = None #save memory by not forming identity
    elif A_type == 'circulant':
        A = (1 / np.sqrt(hparams.problem.num_measurements)) * torch.randn(1, hparams.data.n_input)
    else:
        raise NotImplementedError

    return A

def get_inpaint_mask(hparams):
    image_size = hparams.data.image_size
    inpaint_size = hparams.problem.inpaint_size
    image_shape = hparams.data.image_shape

    margin = (image_size - inpaint_size) // 2
    mask = torch.ones(image_shape)
    mask[margin:margin+inpaint_size, margin:margin+inpaint_size] = 0

    return mask

def get_A_inpaint(hparams):
    mask = get_inpaint_mask(hparams)
    mask = mask.view(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

    return torch.from_numpy(A)

def get_measurements(A, x, hparams):
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian' or A_type == 'inpaint':
        Ax = torch.mm(A, torch.flatten(x, start_dim=1).T).T #[N, m]
    elif A_type == 'superres':
        Ax = F.avg_pool2d(x, hparams.problem.downsample_factor)
    elif A_type == 'identity':
        Ax = torch.nn.Identity(x)
    else:
        raise NotImplementedError #TODO implement circulant!!
    
    return Ax

def get_transpose_measurements(A, vec, hparams):
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian' or A_type == 'inpaint':
        ans = torch.mm(A.T, vec.T).T
    elif A_type == 'superres': #make sure y is in the right shape
        ans = F.interpolate(vec, scale_factor=hparams.problem.downsample_factor)
    elif A_type == 'identity':
        ans = torch.nn.Identity(vec)
    else:
        raise NotImplementedError #TODO implement circulant!!
    
    return ans

def gradient_log_cond_likelihood(c, y, A, x, hparams, scale=1):
    c_type = hparams.outer.hyperparam_type

    Ax = get_measurements(A, x, hparams) #[N, m]
    resid = Ax - y #[N, m]

    grad = 0

    if c_type == 'scalar':
        grad = grad + scale * c * get_transpose_measurements(A, resid, hparams)
    elif c_type == 'vector':
        grad = grad + scale * get_transpose_measurements(A, c * resid, hparams)
    elif c_type == 'matrix':
        vec = torch.mm(torch.mm(c.T, c), resid.T).T #[N, m] 
        grad = grad + scale * get_transpose_measurements(A, vec, hparams)
    else:
        raise NotImplementedError #TODO implement circulant!!
    
    return grad

def log_cond_likelihood_loss(c, y, A, x, hparams, scale=1):
    c_type = hparams.outer.hyperparam_type

    Ax = get_measurements(A, x, hparams) #[N, m]
    resid = Ax - y #[N, m]

    loss = 0

    if c_type == 'scalar':
        loss = loss + scale * c * 0.5 * torch.sum(resid ** 2)
    elif c_type == 'vector':
        loss = loss + scale * 0.5 * torch.sum(c * (resid ** 2))
    elif c_type == 'matrix':
        interior = torch.mm(c, resid.T).T #[N, k]
        loss = loss + scale * 0.5 * torch.sum(interior ** 2)
    else:
        raise NotImplementedError #TODO implement circulant!!

    return loss

def get_meta_loss(x_hat, x_true, hparams):
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.train_loss_type
    ROI = hparams.outer.ROI

    sse = torch.nn.MSELoss(reduction='sum')

    if meas_loss or meta_type != "l2":
        raise NotImplementedError
    
    if ROI:
        ROI = getRectMask(hparams)
        return 0.5 * sse(ROI*x_hat, ROI*x_true)
    else:
        return 0.5 * sse(x_hat, x_true)

def getRectMask(hparams):
    side = hparams.data.image_size
    offsets, hw = hparams.outer.ROI
    h_offset, w_offset = offsets
    height, width = hw

    mask_tensor = torch.zeros(side, side)

    mask_tensor[h_offset:h_offset+height, w_offset:w_offset+width] = 1

    return mask_tensor
