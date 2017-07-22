import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import omniglot

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
        x = self.fc1(x)
        return x

episode_length = 30
episode_width = 5
DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

train_filepath = DATA_FILE_FORMAT % 'train'
trainset = omniglot.OmniglotDataset(train_filepath)
trainloader = trainset.sample_episode_batch(episode_length, episode_width, batch_size=16, N=100000)

test_filepath = DATA_FILE_FORMAT % 'test'
testset = omniglot.OmniglotDataset(test_filepath)
testloader = testset.sample_episode_batch(episode_length, episode_width, batch_size=1, N=50)

'''
#torch.cuda.set_device(1)
net = Net(input_shape=(1,28,28))
net.cuda()

for i, data in enumerate(trainloader, 0):
    # training
    inputs, labels = data
    for x, y in zip(inputs, labels):
        x, y = Variable(x.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

    correct = []
    correct_by_k_shot = dict((k, list()) for k in range(episode_width + 1))
    for i, data in enumerate(testloader, 0):
        x, y = data
        x, y = Variable(x.cuda())
        y_hat = net(x)
        correct.append(torch.mean(torch.eq(y, y_hat)))

        # compute per_shot accuracies
        seen_count = [[0] * episode_width]
        # loop over episode steps
        for yy, yy_hat in zip(y, y_hat):
            count = seen_count[yy % episode_width]
            if count < (episode_width + 1):
                correct_by_k_shot[count].append(torch.eq(yy, yy_hat))
            seen_count[yy % episode_width] += 1

    print("validation overall accuracy {0:f}".format(np.mean(correct)))
    for idx in range(episode_width + 1):
        print("{0:d}-shot: {1:.3f}".format(idx, np.mean(correct_by_k_shot[idx])))
'''
