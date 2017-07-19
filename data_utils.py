import _pickle as pickle
import logging
import os
import subprocess

import numpy as np
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy.ndimage import imread
import tensorflow as tf


REPO_LOCATION = 'https://github.com/brendenlake/omniglot.git'
REPO_DIR = os.path.join(os.getcwd(), 'omniglot')
DATA_DIR = os.path.join(REPO_DIR, 'python')
TRAIN_DIR = os.path.join(DATA_DIR, 'images_background')
TEST_DIR = os.path.join(DATA_DIR, 'images_evaluation')
DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

TEST_ROTATIONS = False  # augment testing data with rotations
IMAGE_ORIGINAL_SIZE = 105
IMAGE_NEW_SIZE = 28

def crawl_directory(directory, augment_with_rotations=False, first_label=0):
  """Crawls data directory and returns stuff."""
  label_idx = first_label
  images = []
  labels = []
  info = []

  # traverse root directory
  for root, _, files in os.walk(directory):
    logging.info('Reading files from %s', root)

    for file_name in files:
      full_file_name = os.path.join(root, file_name)
      img = imread(full_file_name, flatten=True)
      for idx, angle in enumerate([0, 90, 180, 270]):
        if not augment_with_rotations and idx > 0:
          break

        images.append(imrotate(img, angle))
        labels.append(label_idx + idx)
        info.append(full_file_name)

    if len(files) == 20:
      label_idx += 4 if augment_with_rotations else 1
  return images, labels, info


def resize_images(images, new_width, new_height):
  """Resize images to new dimensions."""
  resized_images = np.zeros([images.shape[0], new_width, new_height], dtype=np.float32)

  for idx in range(images.shape[0]):
    resized_images[idx, :, :] = imresize(images[idx, :, :],
                                       [new_width, new_height],
                                       interp='bilinear',
                                       mode=None)
  return resized_images


def write_datafiles(directory, write_file,
                    resize=True, rotate=False,
                    new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                    first_label=0):
  """Load and preprocess images from a directory and write them to a file.

  Args:
    directory: Directory of alphabet sub-directories.
    write_file: Filename to write to.
    resize: Whether to resize the images.
    rotate: Whether to augment the dataset with rotations.
    new_width: New resize width.
    new_height: New resize height.
    first_label: Label to start with.

  Returns:
    Number of new labels created.
  """

  # these are the default sizes for Omniglot:
  imgwidth = IMAGE_ORIGINAL_SIZE
  imgheight = IMAGE_ORIGINAL_SIZE

  logging.info('Reading the data.')
  images, labels, info = crawl_directory(directory, augment_with_rotations=rotate, first_label=first_label)

  images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.bool)
  labels_np = np.zeros([len(labels)], dtype=np.uint32)
  for idx in range(len(images)):
    images_np[idx, :, :] = images[idx]
    labels_np[idx] = labels[idx]

  if resize:
    logging.info('Resizing images.')
    resized_images = resize_images(images_np, new_width, new_height)

    logging.info('Writing resized data in float32 format.')
    data = {'images': resized_images,
            'labels': labels_np,
            'info': info}
    with tf.gfile.GFile(write_file, 'w') as f:
        pickle.dump(data, f)
  else:
    logging.info('Writing original sized data in boolean format.')
    data = {'images': images_np,
            'labels': labels_np,
            'info': info}
    with tf.gfile.GFile(write_file, 'w') as f:
        pickle.dump(data, f)

  return len(np.unique(labels_np))

def maybe_download_data():
  """Download Omniglot repo if it does not exist."""
  if os.path.exists(REPO_DIR):
    logging.info('It appears that Git repo already exists.')
  else:
    logging.info('It appears that Git repo does not exist.')
    logging.info('Cloning now.')

    subprocess.check_output('git clone %s' % REPO_LOCATION, shell=True)

  if os.path.exists(TRAIN_DIR):
    logging.info('It appears that train data has already been unzipped.')
  else:
    logging.info('It appears that train data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TRAIN_DIR, DATA_DIR), shell=True)

  if os.path.exists(TEST_DIR):
    logging.info('It appears that test data has already been unzipped.')
  else:
    logging.info('It appears that test data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TEST_DIR, DATA_DIR), shell=True)

def preprocess_omniglot():
  """Download and prepare raw Omniglot data.

  Downloads the data from GitHub if it does not exist.
  Then load the images, augment with rotations if desired.
  Resize the images and write them to a pickle file.
  """

  maybe_download_data()

  directory = TRAIN_DIR
  write_file = DATA_FILE_FORMAT % 'train'
  num_labels = write_datafiles(directory, write_file, resize=True, rotate=True, new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE)

  directory = TEST_DIR
  write_file = DATA_FILE_FORMAT % 'test'
  write_datafiles(directory, write_file, resize=True, rotate=False, new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE, first_label=num_labels)        

preprocess_omniglot()
