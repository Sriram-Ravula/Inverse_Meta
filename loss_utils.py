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
        A = get_A_inpaint(hparams) 
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
    mask[:, margin:margin+inpaint_size, margin:margin+inpaint_size] = 0

    return mask

def get_A_inpaint(hparams):
    mask = get_inpaint_mask(hparams).numpy()
    mask = mask.view(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

    return torch.from_numpy(A)

def get_measurements(A, x, hparams, efficient_inp=False):
    """
    Efficient_inp uses a mask for element-wise inpainting instead of mm.
    But the dimensions don't play nice with our gradient functions!
    Set=True only if using automatic gradients instead of our gradient functions.
    """
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian':
        Ax = torch.mm(A, torch.flatten(x, start_dim=1).T).T #[N, m]
    elif A_type == 'inpaint' and efficient_inp:
        Ax = get_inpaint_mask(hparams).to(x.device) * x 
    elif A_type == 'inpaint' and not efficient_inp:
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
    A_type = hparams.problem.measurement_type

    #[N, m] (gaussian, inpaint, circulant)
    #[N, C, H//downsample, W//downsample] (superres)
    #[N, C, H, W] for identity
    Ax = get_measurements(A, x, hparams) 
    resid = Ax - y 

    grad = 0.0

    if c_type == 'scalar':
        grad = grad + scale * c * get_transpose_measurements(A, resid, hparams)

    elif c_type == 'vector':
        if A_type == 'superres' or A_type == 'identity':
            c_shaped = c.view(hparams.problem.y_shape)
            grad = grad + scale * get_transpose_measurements(A, c_shaped * resid, hparams)

        else:
            grad = grad + scale * get_transpose_measurements(A, c * resid, hparams)

    elif c_type == 'matrix':
        if A_type == 'superres' or A_type == 'identity':
            y_c, y_h, y_w = hparams.problem.y_shape
            vec = torch.mm(c, resid.flatten(start_dim=1).T).T #[N, k]
            vec = (torch.mm(c.T, vec.T).T).view(-1, y_c, y_h, y_w) #[N, (y_shape)]

        else:
            vec = torch.mm(torch.mm(c.T, c), resid.T).T #[N, m] 

        grad = grad + scale * get_transpose_measurements(A, vec, hparams)

    else:
        raise NotImplementedError
    
    return grad.view(x.shape)

def log_cond_likelihood_loss(c, y, A, x, hparams, scale=1, efficient_inp=False):
    """
    Efficient_inp uses a mask for element-wise inpainting instead of mm.
    But the dimensions don't play nice with our gradient functions!
    Set=True only if using automatic gradients instead of our gradient functions.
    If True, expects a y with the same shape as x.
    """
    if efficient_inp and y.shape != x.shape:
        raise NotImplementedError

    c_type = hparams.outer.hyperparam_type

    #Gaussian or Inpaint + efficient=False - shape [N, m]
    #Inpaint with efficient=True or identity - shape [N, C, H, W]
    #superres - shape [N, C, H//d, W//d]
    #NOTE: Gaussian, inefficient inpaint, superres, identity have no extraneous dimensions  
    #      We can flatten and use normally as size of c accounts for this.
    #NOTE: Efficient Inpaint has extraneous areas of 0 entry that c shape does not account for.
    #      We have to isolate the measurement region to play nice with c.
    Ax = get_measurements(A, x, hparams, efficient_inp) 
    resid = Ax - y 

    loss = 0.0

    if c_type == 'scalar':
        loss = loss + scale * c * 0.5 * torch.sum(resid ** 2)

    elif c_type == 'vector':
        if c_type == 'inpaint' and efficient_inp:
            mask = get_inpaint_mask(hparams)
            kept_inds = (mask.flatten()>0).nonzero(as_tuple=False).flatten()
            loss = loss + scale * 0.5 * torch.sum(c * (resid ** 2).flatten(start_dim=1)[:,kept_inds])

        else:    
            loss = loss + scale * 0.5 * torch.sum(c * (resid ** 2).flatten(start_dim=1))

    elif c_type == 'matrix':
        if c_type == 'inpaint' and efficient_inp:
            mask = get_inpaint_mask(hparams)
            kept_inds = (mask.flatten()>0).nonzero(as_tuple=False).flatten()
            interior = torch.mm(c, resid.flatten(start_dim=1)[:,kept_inds].T).T #[N, k]

        else:
            interior = torch.mm(c, resid.flatten(start_dim=1).T).T #[N, k]

        loss = loss + scale * 0.5 * torch.sum(interior ** 2)

    else:
        raise NotImplementedError #TODO implement circulant!!

    return loss

def meta_loss(x_hat, x_true, hparams):
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.train_loss_type
    ROI = hparams.outer.ROI

    sse = torch.nn.MSELoss(reduction='sum')

    if meas_loss or meta_type != "l2":
        raise NotImplementedError
    
    if ROI:
        ROI = getRectMask(hparams).to(x_hat.device)
        return 0.5 * sse(ROI*x_hat, ROI*x_true)
    else:
        return 0.5 * sse(x_hat, x_true)

def grad_meta_loss(x_hat, x_true, hparams):
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.train_loss_type
    ROI = hparams.outer.ROI

    if meas_loss or meta_type != "l2":
        raise NotImplementedError
    
    if ROI:
        ROI_mat = get_ROI_matrix(hparams).to(x_hat.device)
        vec = torch.mm(ROI_mat, torch.flatten(x_hat - x_true, start_dim=1).T).T #[N, roi_num]
        return (torch.mm(ROI_mat.T, vec.T).T).view(x_hat.shape) 
    else:
        return (x_hat - x_true) #[N, C, H, W]

def get_ROI_matrix(hparams):
    mask = getRectMask(hparams).numpy()
    mask = mask.view(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

    return torch.from_numpy(A)

def getRectMask(hparams):
    shape = hparams.data.image_shape
    offsets, hw = hparams.outer.ROI
    h_offset, w_offset = offsets
    height, width = hw

    mask_tensor = torch.zeros(shape)

    mask_tensor[:, h_offset:h_offset+height, w_offset:w_offset+width] = 1

    return mask_tensor
