import torchvision
import numpy as np
import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import nn
import os
import random

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import argparse

class NCSNv2(torch.nn.Module):
    def __init__(self, hparams, args, c, device=None):
        super().__init__()

        