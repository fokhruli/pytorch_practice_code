# import
import torch



import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running device is {device}....')
# hyperparameter
input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 2
channel = 1
num_class = 10
learning_rate = 0.001
batch_size = 64

# crate cn - net

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc =  nn.Linear(hidden_size*sequence_length, num_class)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
#model = CNN(channel = 1, num_class = 10)

#x = torch.randn(64, 1, 28, 28)
#print(model(x).shape)


num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init net
model = RNN(input_size, hidden_size, num_layers, num_class).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# load network

for e in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            # forward pass
            score = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
        print(f'Got {num_correct} / {num_sample} \
        with accuracy {(float(num_correct)/ float(num_sample))*100:.3f} % ')
    model.train()

check_scores(train_loader, model)
check_scores(test_loader, model)
