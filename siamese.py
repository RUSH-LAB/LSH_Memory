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

for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        x, y = inputs
        print(labels)
