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

def random_index(seed, N):
    """ Args: seed - initial index, N - maximum index
        Return: A random index between [0, N] except for seed
    """
    offset = random.randint(1, N-1)
    idx = (seed + offset) % N 
    assert(seed != idx)
    return idx

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
        self.category_size = len(self.data[0])

    def __len__(self):
        return self.num_categories

    def __getitem__(self, idx):
    	raise NotImplementedError

class TrainSiameseDataset(SiameseDataset):
    def __init__(self, filepath):
        super(SiameseDataset, self).__init__(filepath)

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
        super(SiameseDataset, self).__init__(filepath)
        
    def test_get_item(self, idx):
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
        return self.batch_size * self.N

    def __iter__(self):
        batch_index = 0
        category = random.randint(0, self.data_source.num_categories-1)
        index = random.randint(0, self.data_source.category_size-1)

        for idx in range(self.batch_size * self.rnd):
        	if not sampler_type:
                pos = random.randint(0, len(self.data_source)-1)

            if batch_index < split:
                yield (pos, True)
            else:
                yield (pos, False)

            batch_index += 1
            if batch_index == self.batch_size:
                batch_index = 0
                if sampler_type:
                    category = random.randint(0, self.data_source.num_categories-1)
                    index = random.randint(0, self.data_source.category_size-1)
