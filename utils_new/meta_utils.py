import torch

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
    if verbose:
        print('\n') 
        print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

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