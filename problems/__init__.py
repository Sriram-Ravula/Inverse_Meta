import torch
import numpy as np

from problems.fourier import FourierOperator
from problems.gaussian import GaussianOperator
from problems.identity import IdentityOperator
from problems.inpainting import InpaintingOperator
from problems.superres import SuperresOperator

def get_forward_operator(config):
    if config.problem.measurement_type == 'fourier':
        operator = FourierOperator(config)
    elif config.problem.measurement_type == 'gaussian':
        operator = GaussianOperator(config)
    elif config.problem.measurement_type == 'identity':
        operator = IdentityOperator(config)
    elif config.problem.measurement_type == 'inpaint':
        operator = InpaintingOperator(config)
    elif config.problem.measurement_type == 'superres':
        operator = SuperresOperator(config)
    else:
        raise NotImplementedError
    
    return operator
    