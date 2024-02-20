import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import numpy as np
import jax
import jax.numpy as jnp

import torch.optim

OPTIMS = [
    "Adam", 
    "SGD", 
]

def get_optimizer(optimizer:str, model:nn.Module, hparams:dict):
    '''return optimizer by torch.optim'''
    if optimizer not in OPTIMS:
        raise NotImplementedError("Dataset not found: {}".format(optimizer))
    
    return ...