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
        if hparams.outer.use_autograd:
            A = None
        else:
            A = get_A_inpaint(hparams).float() 
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
    mask = mask.reshape(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0]) #keep rows with 1s in them

    return torch.from_numpy(A)

def get_measurements(A, x, hparams, efficient_inp=False, noisy=False, noise_vars=None):
    """
    Efficient_inp uses a mask for element-wise inpainting instead of mm.
    But the dimensions don't play nice with our gradient functions!
    Set=True only if using automatic gradients instead of our gradient functions.
    """
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian':
        Ax = torch.mm(A, torch.flatten(x, start_dim=1).T).T #[N, m]
    elif A_type == 'inpaint' and efficient_inp:
        Ax = get_inpaint_mask(hparams).to(x.device) * x #[N, C, H, W]
    elif A_type == 'inpaint' and not efficient_inp:
        Ax = torch.mm(A, torch.flatten(x, start_dim=1).T).T #[N, m]
    elif A_type == 'superres':
        Ax = F.avg_pool2d(x, hparams.problem.downsample_factor) #[N, C, H//downsample_factor, W//downsample_factor]
    elif A_type == 'identity':
        I = torch.nn.Identity()
        Ax = I(x)
    else:
        raise NotImplementedError #TODO implement circulant!!
    
    if noisy:
        if hparams.problem.noise_type == 'gaussian':
            noise = torch.randn(Ax.shape, device=Ax.device) * hparams.problem.noise_std
        elif hparams.problem.noise_type == 'gaussian_nonwhite': 
            noise = torch.randn(Ax.shape, device=Ax.device) * hparams.problem.noise_std * noise_vars
        return Ax + noise
    else:
        return Ax

def get_transpose_measurements(A, vec, hparams):
    A_type = hparams.problem.measurement_type

    if A_type == 'gaussian' or A_type == 'inpaint':
        ans = torch.mm(A.T, vec.T).T
    elif A_type == 'superres': #make sure y is in the right shape
        ans = F.interpolate(vec, scale_factor=hparams.problem.downsample_factor)
    elif A_type == 'identity':
        I = torch.nn.Identity()
        ans = I(vec)
    else:
        raise NotImplementedError #TODO implement circulant!!
    
    return ans

def gradient_log_cond_likelihood(c_orig, y, A, x, hparams, scale=1):
    c_type = hparams.outer.hyperparam_type
    A_type = hparams.problem.measurement_type

    if hparams.outer.exp_params:
        c = torch.exp(c_orig)
    else:
        c = c_orig

    #[N, m] (gaussian, inpaint, circulant)
    #[N, C, H//downsample, W//downsample] (superres)
    #[N, C, H, W] for identity
    Ax = get_measurements(A, x, hparams) 
    resid = Ax - y 

    if c_type == 'scalar':
        grad = scale * c * get_transpose_measurements(A, resid, hparams)

    elif c_type == 'vector':
        if A_type == 'superres' or A_type == 'identity':
            c_shaped = c.view(hparams.problem.y_shape)
            grad = scale * get_transpose_measurements(A, c_shaped * resid, hparams)

        else:
            grad = scale * get_transpose_measurements(A, c * resid, hparams)

    elif c_type == 'matrix':
        if A_type == 'superres' or A_type == 'identity':
            y_c, y_h, y_w = hparams.problem.y_shape
            vec = torch.mm(c, resid.flatten(start_dim=1).T).T #[N, k]
            vec = (torch.mm(c.T, vec.T).T).view(-1, y_c, y_h, y_w) #[N, (y_shape)]

        else:
            vec = torch.mm(torch.mm(c.T, c), resid.T).T #[N, m] 

        grad = scale * get_transpose_measurements(A, vec, hparams)

    else:
        raise NotImplementedError
    
    return grad.view(x.shape)

def log_cond_likelihood_loss(c_orig, y, A, x, hparams, scale=1, efficient_inp=False):
    """
    Efficient_inp uses a mask for element-wise inpainting instead of mm.
    But the dimensions don't play nice with our gradient functions!
    Set=True only if using automatic gradients instead of our gradient functions.
    If True, expects a y with the same shape as x.
    """
    if efficient_inp and y.shape != x.shape:
        raise NotImplementedError

    c_type = hparams.outer.hyperparam_type
    A_type = hparams.problem.measurement_type

    if hparams.outer.exp_params:
        c = torch.exp(c_orig)
    else:
        c = c_orig

    #Gaussian or Inpaint + efficient=False - shape [N, m]
    #Inpaint with efficient=True or identity - shape [N, C, H, W]
    #superres - shape [N, C, H//d, W//d]
    #NOTE: Gaussian, inefficient inpaint, superres, identity have no extraneous dimensions  
    #      We can flatten and use normally as size of c accounts for this.
    #NOTE: Efficient Inpaint has extraneous areas of 0 entry that c shape does not account for.
    #      We have to isolate the measurement region to play nice with c.
    Ax = get_measurements(A, x, hparams, efficient_inp) 
    resid = Ax - y 

    if c_type == 'scalar':
        loss = scale * c * 0.5 * torch.sum(resid ** 2)

    elif c_type == 'vector':
        if A_type == 'inpaint' and efficient_inp:
            mask = get_inpaint_mask(hparams)
            kept_inds = (mask.flatten()>0).nonzero(as_tuple=False).flatten()
            loss = scale * 0.5 * torch.sum(c * (resid ** 2).flatten(start_dim=1)[:,kept_inds])

        else:    
            loss = scale * 0.5 * torch.sum(c * (resid ** 2).flatten(start_dim=1))

    elif c_type == 'matrix':
        if A_type == 'inpaint' and efficient_inp:
            mask = get_inpaint_mask(hparams)
            kept_inds = (mask.flatten()>0).nonzero(as_tuple=False).flatten()
            interior = torch.mm(c, resid.flatten(start_dim=1)[:,kept_inds].T).T #[N, k]

        else:
            interior = torch.mm(c, resid.flatten(start_dim=1).T).T #[N, k]

        loss = scale * 0.5 * torch.sum(interior ** 2)

    else:
        raise NotImplementedError #TODO implement circulant!!

    return loss

def simple_likelihood_loss(y, A, x, hparams, efficient_inp=False):
    """
    Returns ||Ax - y||^2 for each image in the batch dimension
    """
    Ax = get_measurements(A, x, hparams, efficient_inp) 
    resid = Ax - y
    resid = resid.flatten(start_dim=1)

    loss = torch.sum(resid**2, dim=[-1]) #shape [N]

    return loss
    

def get_likelihood_grad(c_orig, y, A, x, hparams, scale=1, efficient_inp=False,\
    retain_graph=False, create_graph=False):
    """
    A method for choosing between gradient_log_cond_likelihood (explicitly-formed gradient)
        and log_cond_likelihood_loss with autograd. 
    """
    if hparams.outer.use_autograd:
        grad_flag_x = x.requires_grad
        x.requires_grad_()
        likelihood_grad = torch.autograd.grad(log_cond_likelihood_loss\
                    (c_orig, y, A, x, hparams, scale, efficient_inp), x, retain_graph=retain_graph, create_graph=create_graph)[0]
        x.requires_grad_(grad_flag_x)
    else:
        likelihood_grad = gradient_log_cond_likelihood(c_orig, y, A, x, hparams, scale)
    
    return likelihood_grad

def meta_loss(x_hat, x_true, hparams):
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.meta_loss_type
    ROI = hparams.outer.ROI

    sse = torch.nn.MSELoss(reduction='sum')

    if meas_loss or meta_type != "l2":
        raise NotImplementedError
    
    if ROI:
        ROI = getRectMask(hparams).to(x_hat.device)
        return 0.5 * sse(ROI*x_hat, ROI*x_true)
    else:
        return 0.5 * sse(x_hat, x_true)

def elementwise_meta_loss(x_hat, x_true, hparams):
    """
    Like meta loss, but returns element-wise loss for each image in the batch dimension
    """
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.meta_loss_type
    ROI = hparams.outer.ROI

    if meas_loss or meta_type != "l2":
        raise NotImplementedError

    if ROI:
        ROI = getRectMask(hparams).to(x_hat.device)
        roi_diff = ROI * (x_hat - x_true)
        loss = torch.sum(roi_diff**2, dim=[1,2,3])
    else:
        diff = x_hat - x_true
        loss = torch.sum(diff**2, dim=[1,2,3])
    
    return loss

def grad_meta_loss(x_hat, x_true, hparams):
    meas_loss = hparams.outer.measurement_loss
    meta_type = hparams.outer.meta_loss_type
    ROI = hparams.outer.ROI

    if meas_loss or meta_type != "l2":
        raise NotImplementedError
    
    if ROI:
        ROI_mat = get_ROI_matrix(hparams).to(x_hat.device).float()
        vec = torch.mm(ROI_mat, torch.flatten(x_hat - x_true, start_dim=1).T).T #[N, roi_num]
        return (torch.mm(ROI_mat.T, vec.T).T).view(x_hat.shape) 
    else:
        return (x_hat - x_true) #[N, C, H, W]

def get_meta_grad(x_hat, x_true, hparams, retain_graph=False, create_graph=False):
    """
    A method for choosing between grad_meta_loss (explicitly-formed gradient)
        and meta_loss with autograd. 
    """
    if hparams.outer.use_autograd:
        grad_flag_x = x_hat.requires_grad
        x_hat.requires_grad_()
        meta_grad = torch.autograd.grad(meta_loss\
            (x_hat, x_true, hparams), x_hat, retain_graph=retain_graph, create_graph=create_graph)[0]
        x_hat.requires_grad_(grad_flag_x)
    else:
        meta_grad = grad_meta_loss(x_hat, x_true, hparams)
    
    return meta_grad

def get_ROI_matrix(hparams):
    mask = getRectMask(hparams).numpy()
    mask = mask.reshape(1, -1)
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

def get_loss_dict(y, A, x_hat, x, hparams, efficient_inp=False):
    """
    Calculates and returns a dictionary with the measurement loss and te meta loss
    """
    cur_meta_loss = elementwise_meta_loss(x_hat, x, hparams).detach().cpu().numpy().flatten()
    cur_likelihood_loss = simple_likelihood_loss(y, A, x_hat, hparams, efficient_inp).detach().cpu().numpy().flatten()

    out_dict = {
        'meta_loss': cur_meta_loss,
        'likelihood_loss': cur_likelihood_loss
    }

    return out_dict

def tv_loss(img, weight=1):      
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()    
    return weight * (tv_h + tv_w)
