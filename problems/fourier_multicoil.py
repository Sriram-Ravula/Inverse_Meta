import torch
import torch.fft as torch_fft
import torch.nn as nn

# Multicoil forward operator for MRI
class MulticoilForwardMRI(nn.Module):
    def __init__(self, orientation):
        super(MulticoilForwardMRI, self).__init__()
        self.orientation = orientation
        return

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x

    '''
    Inputs:
     - image = [B, H, W] torch.complex64/128    in image domain
     - maps  = [B, C, H, W] torch.complex64/128 in image domain
     - mask  = [B, W] torch.complex64/128 w/    binary values
    Outputs:
     - ksp_coils = [B, C, H, W] torch.complex64/128 in kspace domain
    '''
    def forward(self, image, maps, mask):
        # Broadcast pointwise multiply
        coils = image[:, None] * maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        if self.orientation == 'vertical':
            # Mask k-space phase encode lines
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == 'horizontal':
            # Mask k-space frequency encode lines
            ksp_coils = ksp_coils * mask[:, None, :, None]
        elif self.orientation == 'random':
            ksp_coils = ksp_coils * mask
        else:
            raise NotImplementedError('mask orientation not supported')

        # Return downsampled k-space
        return ksp_coils

# without mask
class MulticoilForwardMRINoMask(nn.Module):
    def __init__(self, s_maps):
        """
        Args:
            s_maps: [N, C, H, W] complex
        """
        super(MulticoilForwardMRINoMask, self).__init__()

        self.s_maps = s_maps
    
    def ifft(self, x):
        return self._ifft(x)

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x

    def forward(self, image):
        """
        Args:
            image:  [N, 2, H, W] float second channel is (Re, Im)

        Returns:
            ksp_coils: [N, C, H, W] torch.complex64/128 in kspace domain
        """
        #convert to a complex tensor
        x = torch.complex(image[:,0], image[:,1])

        # Broadcast pointwise multiply
        coils = x[:, None] * self.s_maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        # Return k-space
        return ksp_coils