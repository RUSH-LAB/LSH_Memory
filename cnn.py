import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import omniglot
import memory

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        # Constants
        kernel = 3
        pad = int((kernel-1)/2.0)

        ch, row, col = input_shape
        self.conv1 = nn.Conv2d(ch, 64, kernel, padding=(pad, pad))
        self.conv2 = nn.Conv2d(64, 64, kernel, padding=(pad, pad))
        self.conv3 = nn.Conv2d(64, 128, kernel, padding=(pad, pad))
        self.conv4 = nn.Conv2d(128, 128, kernel, padding=(pad, pad))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(row // 4 * col // 4 * 128, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

memory_size = 8192
key_dim = 128
episode_length = 30
episode_width = 5
validation_frequency = 20
DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

train_filepath = DATA_FILE_FORMAT % 'train'
trainset = omniglot.OmniglotDataset(train_filepath)
trainloader = trainset.sample_episode_batch(episode_length, episode_width, batch_size=16, N=100000)

test_filepath = DATA_FILE_FORMAT % 'test'
testset = omniglot.OmniglotDataset(test_filepath)

#torch.cuda.set_device(1)
net = Net(input_shape=(1,28,28))
mem = memory.Memory(memory_size, key_dim)
net.add_module("memory", mem)
net.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-4, eps=1e-4)

cummulative_loss = 0
counter = 0
for i, data in enumerate(trainloader, 0):
    # erase memory before training episode
    mem.erase()
    x, y = data
    for xx, yy in zip(x, y):
        xx_cuda, yy_cuda = Variable(xx.cuda()), Variable(yy.cuda())
        embed = net(xx_cuda)
        yy_hat, softmax_embed, loss = mem.query(embed, yy_cuda, False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cummulative_loss += loss.data[0]
        counter += 1

    if i % validation_frequency == 0:
        # validation
        correct = []
        correct_by_k_shot = dict((k, list()) for k in range(episode_width + 1))
        testloader = testset.sample_episode_batch(episode_length, episode_width, batch_size=1, N=50)
        for data in testloader:
            # erase memory before validation episode
            mem.erase()

            x, y = data
            y_hat = []
            for xx, yy in zip(x, y):
                xx_cuda, yy_cuda = Variable(xx.cuda()), Variable(yy.cuda())
                query = net(xx_cuda)
                yy_hat, embed, loss = mem.query(query, yy_cuda, False)
                correct.append(float(torch.equal(yy_hat, torch.unsqueeze(yy, dim=1))))
                y_hat.append(yy_hat)

            # compute per_shot accuracies
            seen_count = [0 for idx in range(episode_width)]
            # loop over episode steps
            for yy, yy_hat in zip(y, y_hat):
                yy_value = yy[0]
                count = seen_count[yy_value % episode_width]
                if count < (episode_width + 1):
                    correct_by_k_shot[count].append(float(torch.equal(yy_hat, torch.unsqueeze(yy, dim=1))))
                seen_count[yy_value % episode_width] += 1

        print("episode batch: {0:d} average loss: {1:.6f}".format(i, (cummulative_loss / (counter))))
        print("validation overall accuracy {0:f}".format(np.mean(correct)))
        for idx in range(episode_width + 1):
            print("{0:d}-shot: {1:.3f}".format(idx, np.mean(correct_by_k_shot[idx])))
        cummulative_loss = 0
        counter = 0
