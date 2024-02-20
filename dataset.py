import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DATASETS = [
    "CIFAR-10", 
]

def get_dataset(dataset:str):
    '''return dataset as Dataset of pytorch'''
    if dataset not in DATASETS:
        raise NotImplementedError("Dataset not found: {}".format(dataset))
    
    return ...