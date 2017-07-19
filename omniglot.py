import os
import _pickle as pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
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
