import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim


import numpy as np
import jax
import jax.numpy as jnp

import dataset
import model_store


# store hyper parameters
hparams  = {
    "detaset": "CIFAR-10", 
    "optimizer": "SGD", 
    "momentum": 0., 
    "epochs": 10, 
    "train_batch_size": 4, 
    "eval_batch_size": 4, 
    "lr": 1e-3, 
    "checkpoint": 1000, 
}

# avalable GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get dataset
train, test = dataset.get_dataset(hparams["detaset"])
train_loader = torch.utils.data.DataLoader(
                train, 
                batch_size=hparams["train_batch_size"], 
                shuffle=False, 
                num_workers=2)

test_loader = torch.utils.data.DataLoader(
                test, 
                batch_size=hparams["eval_batch_size"], 
                shuffle=False, 
                num_workers=2)


# get model
model = model_store.get_model(hparams).to(device)

# get optimizer
if hparams["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), hparams["lr"], momentum=hparams["momentum"])
elif hparams["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), hparams["lr"])
    
criterion = nn.CrossEntropyLoss()

print("Training Started")
# training phase
for epoch in range(hparams["epochs"]):

    for i, data in enumerate(train_loader, 0):
        images, labes = data
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        
        if i % hparams["checkpoint"] == 0 or i == len(train_loader) - 1:
            print(f"epoch{epoch + 1} - {i + 1}steps loss: {loss}")
            
                    
print("Training Done")

print("Evaluation Started")
# test phase
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        pred_label = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred_label == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
