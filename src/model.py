import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import numpy as np
import jax
import jax.numpy as jnp

MODELS = [
    "MLP", 
    "CNN", 
    "ResNet",  
]

def get_model(model:str, hparams:dict):
    '''return optimizer by torch.optim'''
    if model not in MODELS:
        raise NotImplementedError("Dataset not found: {}".format(model))
    
    return ...

class MLP(nn.Module):
    '''Methods
    - __init__ : initialize components of networks
    - forward : prediction
    '''
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        # define each layer used in networks
        # ex.. self.fc1 = nn.Linear(...)
        
    def forward(self, x):
        # x = self.fc1(x)
        return ...
    
    
class CNN(nn.Module):
    '''Methods
    - __init__ : initialize components of networks
    - forward : prediction
    '''
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        # define each layer used in networks
        # ex.. self.conv1 = nn.Conv2d(...)
        
    def forward(self, x):
        # x = self.conv1(x)
        return ...
    
    
class ResNet(nn.Module):
    # if we have time....