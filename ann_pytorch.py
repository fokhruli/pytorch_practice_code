# import
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# crate fully connected net

class NN(nn.Module):
    def __init__(self, input_size, num_class):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_class)
    def forward(self, x):
        x =  F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = NN(784, 10)

x = torch.randn(64, 784)
print(model(x).shape)
# init device

# hyperparameter

# load data

# init net

# loss and optimizer

# load network

# train and test loss

