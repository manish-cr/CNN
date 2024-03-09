import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import numpy as np
import jax
import jax.numpy as jnp

MODELS = [
    "CNN", 
]

def get_model(hparams:dict):
    model_name = hparams["model"]
    if model_name not in MODELS:
        raise NotImplementedError("Dataset not found: {}".format(model_name))
    
    if model_name == "CNN":
        model_class = CNN(hparams["input_shape"], hparams["n_classes"])
    
    return model_class

    
class CNN(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)        
        return x