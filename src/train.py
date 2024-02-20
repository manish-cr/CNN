import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import numpy as np
import jax
import jax.numpy as jnp

import dataset
import optimizer


# store hyper parameters
hparams  = {
    "detaset": "CIFAR-10", 
    "optimizer": "Adam", 
    "epochs": 10, 
    "batch_size": 128, 
    "lr": 1e-3, 
}

# get dataset
X, y = dataset.get_dataset(hparams["detaset"])

# split data using dataloader
train_x, test_x, train_y, test_y = ..

# get model
model = ...

# get optimizer
optimizer = ...

# training phase
for epoch in range(hparams["epochs"]):
    ...
    
# test phase
