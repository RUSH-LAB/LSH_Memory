import os

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import omniglot

DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')
train_filepath = DATA_FILE_FORMAT % 'train'

N = 1000
batch_size = 32
trainset = omniglot.SiameseDataset(train_filepath)

sampler = omniglot.SiameseSampler(trainset, N, batch_size)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=sampler, num_workers=4)

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        ch, row, col = input_shape
        kernel = 3
        pad = int((kernel-1)/2.0)
        self.conv1 = nn.Conv2d(ch, 64, kernel, padding=(pad, pad))
        self.conv2 = nn.Conv2d(64, 64, kernel, padding=(pad, pad))
        self.conv3 = nn.Conv2d(64, 128, kernel, padding=(pad, pad))
        self.conv4 = nn.Conv2d(128, 128, kernel, padding=(pad, pad))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(row // 4 * col // 4 * 128, 128)
        self.predict = nn.Linear(128, 1)

    def embed(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x, y):
        embed_x = self.embed(x)
        embed_y = self.embed(y)
        l1_distance = torch.abs(embed_x - embed_y)
        result = F.sigmoid(self.predict(l1_distance))
        return result

torch.cuda.set_device(1)
net = Net(input_shape=(1,28,28))
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        x, y = inputs
        x, y, labels = Variable(x.cuda()), Variable(y.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        y_hat = net(x, y)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i == len(trainloader)-1:
            print("[{0:d}, {1:5d}] loss: {2:.3f}".format((epoch+1), (i+1), (running_loss / len(trainloader))))
            running_loss = 0.0

print('Finished Training')
