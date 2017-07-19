import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import omniglot

trainset = omniglot.OmniglotDataset('train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

testset = omniglot.OmniglotDataset('test')
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

classes = 4515

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
        self.softmax = nn.Linear(128, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.softmax(x)
        return x

torch.cuda.set_device(1)
net = Net(input_shape=(1,28,28))
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        y_hat = net(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i == len(trainloader)-1:
            print("[{0:d}, {1:5d}] loss: {2:.3f}".format((epoch+1), (i+1), (running_loss / len(trainloader))))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
for i, data in enumerate(testloader, 0):
    images, labels = data
    images = Variable(images.cuda())
    labels = labels.cuda()
    y_hat = net(images)
    _, predicted = torch.max(y_hat.data, 1)
    total += labels.size(0)
    correct += torch.eq(predicted, labels).sum()

print('Accuracy of the network on the 10000 test images: {0} %'.format(100 * correct / total))
