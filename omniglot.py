import os
import random
import _pickle as pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler as sampler
from torchvision import transforms, utils

def random_index(seed, N):
    """ Args: seed - initial index, N - maximum index
        Return: A random index between [0, N] except for seed
    """
    offset = random.randint(1, N-1)
    idx = (seed + offset) % N 
    assert(seed != idx)
    return idx

class OmniglotDataset(Dataset):
    """Omniglot dataset."""

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
        self.num_categories = len(self.data)
        self.category_size = len(self.data[0])

    def sample_episode_batch(self, N, episode_length, episode_width, batch_size):
        """Generates a random batch for training or validation.

        Structures each element of the batch as an 'episode'.
        Each episode contains episode_length examples and
        episode_width distinct labels.

        Args:
          data: A dictionary mapping label to list of examples.
          episode_length: Number of examples in each episode.
          episode_width: Distinct number of labels in each episode.
          batch_size: Batch size (number of episodes).

        Returns:
          A tuple (x, y) where x is a list of batches of examples
          with size episode_length and y is a list of batches of labels.
          xx = (batch_size, example), yy = (batch_size,)
        """
        for rnd in range(N):
            episodes_x = [list() for _ in range(episode_length)]
            episodes_y = [list() for _ in range(episode_length)]
            assert self.num_categories >= episode_width

            for b in range(batch_size):
                episode_labels = random.sample(self.data.keys(), episode_width)

                # Evenly divide episode_length among episode_width
                remainder = episode_length % episode_width
                remainders = [0] * (episode_width - remainder) + [1] * remainder
                quotient = int((episode_length - remainder) / episode_width)
                episode_x = [random.sample(data[label], r + quotient) for label, r in zip(episode_labels, remainders)]
                assert(quotient+1 <= self.category_size)

                # Arrange episode so that each distinct label is seen before moving to 2nd showing
                # Concatenate class episodes together into single list
                episode = sum([[(example, label_id, example_id) for example_id, example in enumerate(examples_per_label)] for label_id, examples_per_label in enumerate(episode_x)], list())
                random.shuffle(episode)
                episode.sort(key=lambda elem: elem[2])
                assert len(episode) == episode_length

                # During training, the set of labels for each episode are considered distinct
                # The memory is not emptied during each training episode
                for idx in range(episode_length):
                    episodes_x[idx].append(episode[idx][0])
                    episodes_y[idx].append(episode[idx][1] + b * episode_width)

            yield ([torch.from_numpy(np.array(xx)) for xx in episodes_x],
                   [torch.from_numpy(np.array(yy)) for yy in episodes_y])

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
        self.num_categories = len(self.data)
        self.category_size = len(self.data[processed_data['labels'][0]])

    def __len__(self):
        return self.num_categories

    def __getitem__(self, idx):
    	raise NotImplementedError

class TrainSiameseDataset(SiameseDataset):
    def __init__(self, filepath):
        super(TrainSiameseDataset, self).__init__(filepath)

    def __getitem__(self, idx):
        index, same = idx

        if same:
            imageset = self.data[index]
            selected = random.sample(imageset, 2)
            images = [torch.from_numpy(image) for image in selected]
        else:
            left_imageset = self.data[index]
            right_imageset = self.data[random_index(index, self.num_categories)]
            left_img = random.sample(left_imageset, 1)
            right_img = random.sample(right_imageset, 1)
            images = [torch.from_numpy(image) for image in (left_img + right_img)]

        label = int(same)
        sample = [images, label]
        return sample

class TestSiameseDataset(SiameseDataset):
    def __init__(self, filepath):
        super(TestSiameseDataset, self).__init__(filepath)
        
    def __getitem__(self, idx):
        """ Args: [test_image, same] = idx 
            test_image = (test_category, test_category_image)
            same (bool) = if support image comes from the same category
        """
        test_id, same = idx
        category, index = test_id
        test_img = self.data[category][index]

        support_idx = random_index(index, self.category_size)
        if same:
            support_img = self.data[category][support_idx]
        else:
            support_category = random_index(category, self.num_categories)
            support_img = self.data[support_category][support_idx]

        selected = (test_img, support_img)
        images = [torch.from_numpy(image) for image in selected]
        label = int(same)
        sample = [images, label]
        return sample

class SiameseSampler(sampler.Sampler):
    """Samples elements for Siamese Network Training."""

    def __init__(self, data_source, rnd, batch_size, sampler_type):
        """ Args: classes - number of classes in dataset
                  rnd - number of iterations
                  batch_size - size of batch
                  sampler_type - (test = 1) OR (train = 0)
                  split (int) - index to switch from same (label=1) to different (label=0)
        """
        self.data_source = data_source
        self.rnd = rnd
        self.batch_size = batch_size
        self.sampler_type = sampler_type
        self.split = 1 if sampler_type else int(batch_size/2)

    def __len__(self):
        return self.batch_size * self.rnd

    def __iter__(self):
        if self.sampler_type:
            pos = self.generate_test()

        batch_index = 0
        for idx in range(self.batch_size * self.rnd):
            if not self.sampler_type:
                pos = random.randint(0, len(self.data_source)-1)

            if batch_index < self.split:
                yield (pos, True)
            else:
                yield (pos, False)

            batch_index += 1
            if batch_index == self.batch_size:
                batch_index = 0
                if self.sampler_type:
                    pos = self.generate_test()

    def generate_test(self):
        category = random.randint(0, self.data_source.num_categories-1)
        index = random.randint(0, self.data_source.category_size-1)
        return (category, index)
