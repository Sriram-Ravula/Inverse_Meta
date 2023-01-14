import torch
import torch.nn as nn
import torch.nn.functional as F

class Probabilistic_Mask(torch.nn.Module):
    def __init__(self, hparams, args):
        super().__init__()

        self.hparams = hparams
        self.args = args

        #(1) Initialise
        #In config, 