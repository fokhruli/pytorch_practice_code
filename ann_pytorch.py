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
#model = NN(784, 10)

#x = torch.randn(64, 784)
#print(model(x).shape)

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                                                               "")
# hyperparameter
input_size = 784
num_class = 10
learning_rate = 0.001
batch_size = 64
epoch = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init net
model = NN(input_size=input_size, num_class=num_class).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# load network
for e in range(epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # reshape
        data = data.reshape(batch_size, -1)
        # forward pass
        score = model(data)
        loss = criterion(score, targets)
        #backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

# train and test loss

def check_scores():
    pass