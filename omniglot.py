import os
import random
import _pickle as pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler as sampler
from torchvision import transforms, utils

DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

class OmniglotDataset(Dataset):
    """Omniglot dataset."""

    def __init__(self, data_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(DATA_FILE_FORMAT % data_file, "rb") as f:
            processed_data = pickle.load(f)
        self.images = np.vstack([np.expand_dims(np.expand_dims(image, axis=0), axis=0) for image in processed_data['images']])
        self.images = self.images.astype('float32')
        self.images /= 255.0

        self.labels = processed_data['labels']
        self.labels = self.labels.astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        label = self.labels[idx]
        sample = [torch.from_numpy(image), label]
        return sample

class SiameseDataset(Dataset):
    """Siamese Dataset dataset."""

    def __init__(self, filepath):
        """
        Args:
            filepath (string): path to data file
            Data format - list of characters, list of images, (row, col, ch) numpy array normalized between (0.0, 1.0)
            Omniglot dataset - Each language contains a set of characters; Each character is defined by 20 different images
        """
        with open(filepath, "rb") as f:
            processed_data = pickle.load(f)

        self.data = dict()
        for image, label in zip(processed_data['images'], processed_data['labels']):
            if label not in self.data:
                self.data[label] = list()
            img = np.expand_dims(image, axis=0).astype('float32')
            img /= 255.0
            self.data[label].append(img)

    def __len__(self):
        return len(self.data)

    def random_index(self, seed):
        """ Args: seed - initial index
            Return: A random index between [0, N] except for seed
        """
        offset = random.randint(1, len(self.data)-1)
        idx = (seed + offset) % len(self.data)
        assert(seed != idx)
        return idx

    def __getitem__(self, idx):
        index, same = idx
        label = int(same)

        if same:
            imageset = self.data[index]
            selected = random.sample(imageset, 2)
            images = [torch.from_numpy(image) for image in selected]
        else:
            left_imageset = self.data[index]
            right_imageset = self.data[self.random_index(index)]
            left_img = random.sample(left_imageset, 1)
            right_img = random.sample(right_imageset, 1)
            images = [torch.from_numpy(image) for image in (left_img + right_img)]

        sample = [images, label]
        return sample

class SiameseSampler(sampler.Sampler):
    """Samples elements for Siamese Network Training."""

    def __init__(self, data_source, N, batch_size):
        """ Args: classes - number of classes in dataset
                  N - number of batches
                  batch_size - size of batch
        """
        self.data_source = data_source
        self.N = N
        self.batch_size = batch_size

    def __iter__(self):
        batch_index = 0
        for idx in range(self.batch_size * self.N):
            pos = random.randint(0, len(self.data_source)-1)
            if batch_index < int(self.batch_size/2):
                yield (pos, True)
            else:
                yield (pos, False)

            batch_index += 1
            if batch_index == self.batch_size:
                batch_index = 0

    def __len__(self):
        return self.batch_size * self.N
