import torch
import numpy as np
import loss_utils

def hessian_vector_product(x, cond_log_grad, model, sigma_idx, vec, hparams):
    device = hparams.device
    finite_difference = hparams.outer.finite_difference
    net = hparams.net

    if finite_difference or net != "ncsnv2":
        raise NotImplementedError #TODO implement finite difference and other models!
    
    labels = torch.ones(x.shape[0], device=device) * sigma_idx
    labels = labels.long()

    full_grad = cond_log_grad - model(x, labels) #full gradient of loss w.r.t. x

    h_func = torch.sum(full_grad * vec) #v.T (dL/dx)
    hvp = torch.autograd.grad(h_func, x)[0] #[N, C, H, W]

    return hvp

def cross_hessian_vector_product(c, cond_log_grad, vec, hparams):
    net = hparams.net

    if net != "ncsnv2":
        raise NotImplementedError #TODO implement other models!

    h_func = torch.sum(cond_log_grad * vec) #v.T (dL/dx)

    hvp = torch.autograd.grad(h_func, c)[0] #[N, C, H, W]

    return hvp

def Ax(x, cond_log_grad, model, sigma_idx, hparams):
    damping = hparams.outer.cg_damping

    def hvp_evaluator(vec):
        undamped = hessian_vector_product(x, cond_log_grad, model, sigma_idx, vec, hparams)
        
        return damping * vec + undamped #Hv --> (aI + H)v = av + Hv
    
    return hvp_evaluator

def cg_solver(f_Ax, b, hparams, x_init=None):
    device = hparams.device
    residual_tol = hparams.outer.cg_tol
    cg_iters = hparams.outer.cg_iters
    verbose = hparams.outer.verbose

    if verbose:
        verbose = hparams.outer.cg_verbose

    x = torch.zeros(b.shape, device=device) if x_init is None else x_init

    r = b - f_Ax(x) 
    p = r.clone()

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if verbose and i % verbose == 0:
            obj_fn = 0.5 * torch.sum(x * f_Ax(x)) - 0.5 * torch.sum(b * x)
            norm_x = torch.norm(x)
            norm_r = torch.norm(r)
            print(fmtstr % (i, norm_r, norm_x, obj_fn))
        
        rsold = torch.sum(r ** 2) 
        Ap = f_Ax(p)
        alpha = rsold / (torch.sum(p * Ap))

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = torch.sum(r ** 2) 

        if torch.sqrt(rsnew) < residual_tol:
            if verbose:
                print("Early CG termination due to small residual value")
            break

        p = r + (rsnew / rsold) * p

    if verbose:
        obj_fn = 0.5 * torch.sum(x * f_Ax(x)) - 0.5 * torch.sum(b * x)
        norm_x = torch.norm(x)
        norm_r = torch.norm(r)
        print(fmtstr % (i+1, norm_r, norm_x, obj_fn))
        print("\n")

    return x  

def SGLD_inverse(c, y, A, x_mod, model, sigmas, hparams):
    T = hparams.inner.T
    step_lr = hparams.inner.lr
    decimate = hparams.inner.decimation_factor
    add_noise = True if hparams.inner.alg == 'langevin' else False
    verbose = hparams.outer.verbose
    if verbose:
        verbose = hparams.inner.verbose
    if hparams.net != "ncsnv2":
        raise NotImplementedError #TODO implement finite difference and other models!
  
    #TODO alter this for non-implicit meta
    grad_flag = x_mod.requires_grad
    x_mod.requires_grad_(False) 

    if decimate:
        used_levels = get_decimated_sigmas(len(sigmas), hparams)
        num_used_levels = len(used_levels)
    else:
        num_used_levels = len(sigmas)
        used_levels = np.arange(len(sigmas))

    global_step = 0

    #iterate over noise level index
    for t in used_levels:
        sigma = sigmas[t]

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * t
        labels = labels.long()

        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for s in range(T):
            prior_grad = model(x_mod, labels)
            likelihood_grad = loss_utils.gradient_log_cond_likelihood(c, y, A, x_mod, hparams, scale=1/(sigma**2))

            grad = prior_grad - likelihood_grad

            if add_noise:
                noise = torch.randn_like(x_mod)
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)
            else:
                x_mod = x_mod + step_size * grad

            #logging
            if verbose and (global_step % verbose == 0 or global_step == T*num_used_levels - 1):
                prior_grad_norm = torch.norm(prior_grad.view(prior_grad.shape[0], -1), dim=-1).mean()
                likelihood_grad_norm = torch.norm(likelihood_grad.view(likelihood_grad.shape[0], -1), dim=-1).mean()
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()

                print("Noise Level: {}, Step Size: {:.3f}, Prior Grad Norm: {:.3f}, Likelihood Grad Norm: {:.3f}, Total Grad Norm: {:.3f}".format(
                    t, step_size, prior_grad_norm.item(), likelihood_grad_norm.item(), grad_norm.item()))
      
            global_step += 1
  
    x_mod.requires_grad_(grad_flag)

    return x_mod

def get_decimated_sigmas(L, hparams):
    decimate = hparams.inner.decimation_factor
    decimate_type = hparams.inner.decimation_type

    num_used_levels = L // decimate 

    #geometrically-spaced entries biased towards later noise levels
    if decimate_type == 'log_last':
        used_levels = np.ceil(-np.geomspace(start=L, stop=1, num=num_used_levels)+L).astype(np.long)
    
    #geometrically-spaced entries biased towards earlier noise levels
    elif decimate_type == 'log_first':
        used_levels = (np.geomspace(start=1, stop=L, num=num_used_levels)-1).astype(np.long)

    #grab just the last few noise levels
    elif decimate_type == 'last':
        used_levels = np.arange(L)[-num_used_levels:]
    
    #grab just the first few noise levels
    elif decimate_type == 'first':
        used_levels = np.arange(L)[:num_used_levels]

    #grab equally-spaced levels
    elif decimate_type == 'linear':
        used_levels = np.ceil(np.linspace(start=0, stop=L-1, num=num_used_levels)).astype(np.long)
    
    else:
        raise NotImplementedError

    return used_levels
