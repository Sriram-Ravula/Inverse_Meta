from problems.fourier_multicoil import MulticoilForwardMRINoMask
import torch
import numpy as np
from tqdm import tqdm

from utils_new.cg import ZConjGrad, get_Aop_fun, get_cg_rhs
from utils_new.helpers import unnormalize, get_mvue_torch, get_min_max, normalize, complex_to_real, real_to_complex


def single_step_posterior_estimate(net, x_t, sigma_t, FSx, P, S, likelihood_step_size):
    """
    Performs a single-step reconstruction using the posterior sampling version of Tweedie's formula.
    Uses the approximation: E[x_0 | x_t, y] = x^ - c * (d / dx^)||PFSx^ - y||^2, where x^ = E[x_0 | x_t].

    Args:
        net (Torch Module): The diffusion network, assumed to be a denoiser (i.e. net(x + noise) = x).
                            Assumed to be an EDM network (Karras et al., 2022).
        x_t ([N, 2, H, W] real-valued Torch Tensor): The noised sample. 
                                                     The second axis holds the real and imaginary components.
        sigma_t ([N, 1, 1, 1] Torch Tensor): Noise standard deviation at time t.
        FSx ([N, C, H, W] complex Torch Tensor): Fully-sampled ground truth k-space with C coil measurements.
        P ([N, 1, H, W] real-valued Torch Tensor): The sampling pattern, entries should be in [0, 1].
        S ([N, C, H, W] complex Torch Tensor): Sensitivity maps for each of C coils for each of the N samples.
        likelihood_step_size (float): Step size parameter for the likelihood term.
    
    Returns:
        x_hat ([N, 2, H, W] real-valued Torch Tensor): Single-step posterior reconstruction E[x_0 | x_t, y].
    """
    #(0) Setup
    device = x_t.device
    
    FS = MulticoilForwardMRINoMask(S) #[N, 2, H, W] float --> [N, C, H, W] complex
    
    class_labels = None
    if net.label_dim:
        class_labels = torch.zeros((x_t.shape[0], net.label_dim), device=device) #[N, label_dim]
    
    with torch.no_grad():
        y = P * FSx #[N, C, H, W] complex, the undersampled multi-coil k-space measurements
        x_hat_mvue = get_mvue_torch(y, S)
        norm_mins, norm_maxes = get_min_max(x_hat_mvue)
    
    #(1) Get the unconditional denoised estimate and unscale properly
    x_hat_0 = net(x_t, sigma_t, class_labels)
    x_hat_0.requires_grad_()
    x_hat_0_unscaled = unnormalize(x_hat_0, norm_mins, norm_maxes)
    
    #(2) Calculate the likelihood gradient
    residual = P * (FS(x_hat_0_unscaled) - FSx)
    sse = torch.sum(torch.square(torch.abs(residual)))
    likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat_0, create_graph=True)[0] #create a graph to calculate loss gradients
    
    #(3) Create the final posterior mean prediction and unnormalize properly
    x_hat = x_hat_0 - likelihood_step_size * likelihood_score
    
    return unnormalize(x_hat, norm_mins, norm_maxes)

def get_noise_schedule(steps, sigma_max, sigma_min, rho, net, device):
    """
    Generates a [steps + 1] torch tensor with sigma values for reverse diffusion
        in descending order (final entry is always 0).
    """
    if steps == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float64, device=device)
    
    step_indices = torch.arange(steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    return t_steps

def MRI_diffusion_sampling(net, x_init, t_steps, FSx, P, S, alg_type,
                            S_churn=0., S_min=0., S_max=float('inf'), S_noise=1.,
                            gradient_update_steps=0, **kwargs):
    """
    Performs conditional sampling for solving MRI inverse problems using a diffusion-based model.
    
    Args:
        net (Torch Module): The diffusion network, assumed to be a denoiser (i.e. net(x + noise) = x).
                            Assumed to be an EDM network (Karras et al., 2022).
        x_init ([N, 2, H, W] real-valued Torch Tensor): The initialisation for the reverse diffusion process. 
                                                        The second axis holds the real and imaginary components. 
                                                        This should be normalised and noised appropriately before passing to MRI_diffusion_sampling().
        t_steps ([steps + 1] Torch Tensor): Sigma values for the reverse diffusion process in descending order.
                                            Length (steps + 1) means last entry should be 0.
                                            MRI_diffusion_sampling() performs steps number of iterations.
        FSx ([N, C, H, W] complex Torch Tensor): Fully-sampled ground truth k-space with C coil measurements.
        P ([N, 1, H, W] real-valued Torch Tensor): The sampling pattern, entries should be in [0, 1].
        S ([N, C, H, W] complex Torch Tensor): Sensitivity maps for each of C coils for each of the N samples.
        alg_type (string): The sampling algorithm to perform, from ["dps", "shallow_dps", "cg", "repaint"].
        gradient_update_steps (int >= 0): Number of sampling steps to track on the computational graph.
                                          Setting to 0 means we don't want to propagate gradients. 
                                          Will track the steps at the end of sampling, e.g. if =5 then the last
                                            5 sampling steps will be added to the computational graph.
    
    Returns:
        x_hat: ([N, 2, H, W] real-valued Torch Tensor): Output of the reverse diffusion process.
    """
    #(0) Setup
    device = x_init.device
    
    FS = MulticoilForwardMRINoMask(S) #[N, 2, H, W] float --> [N, C, H, W] complex
    
    class_labels = None
    if net.label_dim:
        class_labels = torch.zeros((x_init.shape[0], net.label_dim), device=device) #[N, label_dim]
    
    with torch.no_grad():
        y = P * FSx #[N, C, H, W] complex, the undersampled multi-coil k-space measurements
        x_hat_mvue = get_mvue_torch(y, S)
        norm_mins, norm_maxes = get_min_max(x_hat_mvue)
        
    total_steps = torch.numel(t_steps) - 1
    grad_flag = False
    
    #(1) Sampling Loop
    x_next = x_init
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., T-1
        if (total_steps - i) == gradient_update_steps:
            grad_flag = True
            
        x_cur = x_next
        
        # (1a) Increase noise temporarily (for non-DDIM sampling).
        gamma = min(S_churn / (t_steps.numel() - 1), np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0 
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
        x_t_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur) 
        if alg_type == "dps":
            x_t_hat.requires_grad_()
        
        # (1b) Get the denoised network output
        x_denoised = net(x_t_hat, t_hat, class_labels)
        if alg_type == "shallow_dps":
            x_denoised.requires_grad_()
        x_denoised_unscaled = unnormalize(x_denoised, norm_mins, norm_maxes)
        
        # (1c) grab the negative score term d_cur
        if "dps" in alg_type:
            d_cur = (x_t_hat - x_denoised) / t_hat
        elif alg_type == "cg":
            x_cg_init = real_to_complex(x_denoised_unscaled) #[N, 2, H, W] real --> [N, H, W] complex
            Aop_fun = get_Aop_fun(P, S)
            cg_rhs = get_cg_rhs(P, S, FSx, kwargs['cg_lambda'], x_cg_init)
            
            CG_Runner = ZConjGrad(cg_rhs, Aop_fun, kwargs['cg_max_iter'], kwargs['cg_lambda'], kwargs['cg_eps'], False)
            x_cg = CG_Runner.forward(x_cg_init)
            
            x_cg_real = complex_to_real(x_cg)
            x_cg_real_scaled = normalize(x_cg_real, norm_mins, norm_maxes)
            
            d_cur = (x_t_hat - x_cg_real_scaled) / t_hat
        elif alg_type == "repaint":
            y_repaint = (1 - P) * FS(x_denoised_unscaled) + P * FSx
            x_repaint = get_mvue_torch(y_repaint, S)
            x_repaint_scaled = normalize(x_repaint, norm_mins, norm_maxes)
            
            d_cur = (x_t_hat - x_repaint_scaled) / t_hat
        else:
            raise NotImplementedError("Given alg_type not supported!")
        
        # (1d) grab the likelihood score
        if "dps" in alg_type:
            residual = P * (FS(x_denoised_unscaled) - FSx)
            sse_per_samp = torch.sum(torch.square(torch.abs(residual)), dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
            sse = torch.sum(sse_per_samp)
            
            if alg_type == "shallow_dps":
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_denoised, create_graph=grad_flag)[0] 
            else:
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_t_hat, create_graph=grad_flag)[0] 
            
            if kwargs['normalize_grad']:
                likelihood_score = (kwargs['likelihood_step_size'] / torch.sqrt(sse_per_samp).detach()) * likelihood_score 
            else:
                likelihood_score = kwargs['likelihood_step_size'] * likelihood_score
        else:
            likelihood_score = 0. 
        
        # (1e) Take an Euler step using the gradient d_cur and the likelihood score
        x_next = x_t_hat + (t_next - t_hat) * d_cur - likelihood_score
        if not grad_flag:
            x_next = x_next.detach()
    
    return unnormalize(x_next, norm_mins, norm_maxes)
