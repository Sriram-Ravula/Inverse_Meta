import torch
import numpy as np
import loss_utils

def hessian_vector_product(x, jacobian, vec, hparams, retain_graph=False):
    """
    Calculates a Hessian-vector product with the given first derivative (Jacobian).

    Args:
        x: Hessian is taken w.r.t. this variable. Jacobian must be a function of x.
           Torch tensor.
        jacobian: The first derivative of the function whose Hessian we want.
                  Must be a leaf tensor with root at x.
                  Torch tensor with dimension [N, C, H, W].
        vec: The vector in the Hessian-vector product.
             Torch tensor with dimension that can broadcast with jacobian dimension.
        hparams: Experimental parameters.
        retain_graph: Whether to keep the computation graphs of x, jacobian, and vec intact.
                      Useful when e.g. we need to re-use the same Jacobian multiple times
                        like in conjugate gradient during implicit meta methods.

    Returns:
        hvp: Hessian of the loss function w.r.t. x, right-multiplied with vec.
             Torch Tensor with same shape as x.
    """
    finite_difference = hparams.outer.finite_difference
    net = hparams.net.model

    if finite_difference or net != "ncsnv2":
        raise NotImplementedError #TODO implement finite difference and other models!

    h_func = torch.sum(jacobian * vec) #v.T (dL/dx)

    hvp = torch.autograd.grad(h_func, x, retain_graph=retain_graph)[0]

    return hvp

def Ax(x, jacobian, hparams, retain_graph=False):
    """
    Helper function that returns a hessian-vector product evaluator.
    Plug into CG optimization to evaluate Ax (where here A is the Hessian and x is vector).
    NOTE: only works when x and vector have the same shape
    """
    damping = hparams.outer.cg_damping

    def hvp_evaluator(vec):
        undamped = hessian_vector_product(x, jacobian, vec, hparams, retain_graph=retain_graph)

        return damping * vec + undamped #Hv --> (aI + H)v = av + Hv

    return hvp_evaluator

def cg_solver(f_Ax, b, hparams, x_init=None):
    """
    Solve the system Ax = b for x using the conjugate gradient method.
    Used in implicit methods to solve for inverse hessian-meta gradient product
    where the Hessian is of the inner loss w.r.t. x and meta gradient is w.r.t. x.

    Args:
        f_Ax:   A function that takes x as input and calculates Ax.
                Usually Hessian-vector product where Hessian is of inner loss w.r.t. x.
                Output is Torch tensor with shape [N, C, H, W].
        b:  A target quantity. Usually gradient of meta-objective w.r.t. x.
            Torch tensor with shape [N, C, H, W].
        hparams: Experimental parameters.
        x_init: (Optional) an initial value for the variable we are solving for.
                If set = None, x is initialized as all 0 tensor. Default value = None.

    Returns:
        x:  The solution to Ax = b as found by conjugate gradient algorithm.
            Torch tensor with shape [N, C, H, W].
    """

    residual_tol = hparams.outer.cg_tol
    cg_iters = hparams.outer.cg_iters
    verbose = hparams.outer.verbose

    if verbose:
        verbose = hparams.outer.cg_verbose if hparams.outer.cg_verbose > 0 else False

    if cg_iters < 1:
        return b.clone()

    x = torch.zeros(b.shape, device=b.device) if x_init is None else x_init

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
                print("\nEarly CG termination due to small residual value\n")
                verbose=False
            break

        p = r + (rsnew / rsold) * p

    if verbose:
        obj_fn = 0.5 * torch.sum(x * f_Ax(x)) - 0.5 * torch.sum(b * x)
        norm_x = torch.norm(x)
        norm_r = torch.norm(r)
        print(fmtstr % (i+1, norm_r, norm_x, obj_fn))
        print("\n")

    return x

def SGLD_inverse(c, y, A, x_mod, model, sigmas, hparams, s_maps=None, eval=False):
    T = hparams.inner.T
    step_lr = hparams.inner.lr
    decimate = hparams.inner.decimation_factor if hparams.inner.decimation_factor > 0 else False
    add_noise = True if hparams.inner.alg == 'langevin' else False
    maml_use_last = hparams.outer.maml_use_last
    use_autograd = hparams.outer.auto_cond_log
    verbose = hparams.outer.verbose

    if verbose:
        verbose = hparams.inner.verbose if hparams.inner.verbose > 0 else False

    if decimate:
        used_levels = get_decimated_sigmas(len(sigmas), hparams)
        total_steps = len(used_levels) * T
    else:
        total_steps = len(sigmas) * T
        used_levels = np.arange(len(sigmas))

    if maml_use_last not in np.arange(start=1, stop=total_steps) or hparams.outer.meta_type != 'maml':
        maml_use_last = False

    if hparams.outer.meta_type == 'maml' and (not maml_use_last or maml_use_last==total_steps):
        create_graph = True
    else:
        create_graph = False

    if not create_graph or eval:
        grad_flag_x = x_mod.requires_grad
        x_mod.requires_grad_(False)
        grad_flag_c = c.requires_grad
        c.requires_grad_(False)

    if hparams.outer.auto_cond_log and hparams.outer.hyperparam_type == 'inpaint':
        efficient_inp = True
    else:
        efficient_inp = False

    fmtstr = "%10i %10.3g %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s %10s"
    if verbose: print(titlestr % ("Noise Level", "Meas Loss", "Score Norm", "Meas Grad Norm", "Total Grad Norm"))

    step_num = 0

    #iterate over noise level index
    for t in used_levels:
        sigma = sigmas[t]

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * t
        labels = labels.long()

        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for s in range(T):
            if not eval and not create_graph and hparams.outer.meta_type == 'maml' and (total_steps - step_num) == maml_use_last:
                create_graph = True
                x_mod.requires_grad_()
                c.requires_grad_()

            prior_grad = model(x_mod, labels)

            if not create_graph and use_autograd:
                x_mod.requires_grad_()
                likelihood_loss = loss_utils.log_cond_likelihood_loss(c, y, A, x_mod, hparams, scale=1/(sigma**2), efficient_inp=efficient_inp)
                likelihood_grad = torch.autograd.grad(likelihood_loss, x_mod, create_graph=create_graph)[0]
                x_mod.requires_grad_(False)
            elif use_autograd:
                likelihood_loss = loss_utils.log_cond_likelihood_loss(c, y, A, x_mod, hparams, scale=1/(sigma**2), efficient_inp=efficient_inp)
                likelihood_grad = torch.autograd.grad(likelihood_loss, x_mod, create_graph=create_graph)[0]
            else:
                likelihood_grad = loss_utils.gradient_log_cond_likelihood(
                                                        c, y, A,
                                                        x_mod,
                                                        hparams,
                                                        scale=1/(sigma**2),
                                                        s_maps=s_maps)

            grad = prior_grad - likelihood_grad

            if add_noise:
                noise = torch.randn_like(x_mod)
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)
            else:
                x_mod = x_mod + step_size * grad

            if verbose and (step_num % verbose == 0 or step_num == total_steps - 1):
                with torch.no_grad():
                    prior_grad_norm = torch.norm(prior_grad.view(prior_grad.shape[0], -1), dim=-1).mean().item()
                    likelihood_grad_norm = torch.norm(likelihood_grad.view(likelihood_grad.shape[0], -1), dim=-1).mean().item()
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean().item()
                    likelihood_loss = loss_utils.log_cond_likelihood_loss(c, y, A, x_mod, hparams, scale=1, efficient_inp=efficient_inp).item()
                    likelihood_loss /= x_mod.shape[0]

                    print(fmtstr % (t, likelihood_loss, prior_grad_norm, likelihood_grad_norm, grad_norm))

            step_num += 1

    x_mod = torch.clamp(x_mod, 0.0, 1.0)

    if not create_graph:
        x_mod.requires_grad_(grad_flag_x)
        c.requires_grad_(grad_flag_c)

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
