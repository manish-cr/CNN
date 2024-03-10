import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


DATASETS = [
    "CIFAR-10", 
]

def get_dataset(dataset:str, dir="./data"):
    '''return dataset as a Dataset class of pytorch'''
    if dataset not in DATASETS:
        raise NotImplementedError("Dataset not found: {}".format(dataset))
    
    if dataset == "CIFAR-10":
        transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train =  datasets.CIFAR10(
            root=dir, 
            train=True, 
            download=True, 
            transform=transform)
        
        test =  datasets.CIFAR10(
            root=dir, 
            train=False, 
            download=True, 
            transform=transform)
        
    return train, test

