# import
import torch



import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# crate cn - net

class CNN(nn.Module):
    def __init__(self, channel = 1, num_class = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_class)
    def forward(self, x):
        x =  F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
#model = CNN(channel = 1, num_class = 10)

#x = torch.randn(64, 1, 28, 28)
#print(model(x).shape)

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running device is {device}....')
# hyperparameter
channel = 1
num_class = 10
learning_rate = 0.001
batch_size = 64
epoch = 5

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init net
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# load network

for e in range(epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        score = model(data)
        loss = criterion(score, targets)
        #backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

# train and test loss

def check_scores(data_loader, model):

    if data_loader.dataset.train:
        print("checking accuracy on training set")
    else:
        print("checking accuracy on testing set")
    num_correct = 0
    num_sample = 0
    model.eval()


    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # forward pass
            score = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
        print(f'Got {num_correct} / {num_sample} with accuracy {(float(num_correct)/ float(num_sample))*100:.3f} % ')
    model.train()

check_scores(train_loader, model)
check_scores(test_loader, model)
