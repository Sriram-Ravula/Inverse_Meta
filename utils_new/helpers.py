import torch
import torch.fft as torch_fft

def ifft(x):
    """
    Centered, orthogonal ifft in torch. input is [N, C, H, W] complex. 
    """
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

def fft(x):
    """
    Centered, orthogonal fft in torch. input is [N, C, H, W] complex. 
    """
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def normalize(x, x_min, x_max):
    """
    Scales x to appx [-1, 1]
    """
    out = (x - x_min) / (x_max - x_min)
    return 2*out - 1

def unnormalize(x, x_min, x_max):
    """
    Takes input in appx [-1,1] and unscales it
    """
    out = (x + 1) / 2
    return out * (x_max - x_min) + x_min

def real_to_complex(x):
    """
    Takes an [N, 2, H, W] real-valued tensor and converts it to a [N, H, W] complex tensor.
    """
    return torch.complex(x[:,0], x[:,1])

def complex_to_real(x):
    """
    Converts [N, H, W] complex tensor to a [N, 2, H, W] real-valued tensor.
    """
    return torch.permute(torch.view_as_real(x), (0, 3, 1, 2))

def get_mvue_torch(y, s_maps):
    """
    Given multi-coil measurements and coil sensitivity maps, return the MVUE.
    
    Args:
        y: Undersampled multi-coil measurements PFSx. [N, C, H, W] complex tensor.
        s_maps: Coil sensitivity maps S. [N, C, H, W] complex tensor. 
    
    Returns:
        estimated_mvue: [N, H, W, 2] real-valued tensor.
    """
    estimated_mvue = torch.sum(ifft(y) * torch.conj(s_maps), axis=1) / torch.sqrt(torch.sum(torch.square(torch.abs(s_maps)), axis=1))
    return torch.view_as_real(estimated_mvue)
