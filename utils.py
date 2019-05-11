import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tfs

# defining the hyperparameters.
epochs = 30
lr = 0.001
batch_size = 32
num_workers = 4

_class = pd.read_csv('./CUB_200_2011/classes.txt', sep=' ', names=['id','label'])
classes = {str(i):str(cl)[4:] for i, cl in zip(list(_class.id), list(_class.label))}


# defining the class to load the images.
class BirdDataset(Dataset):

  def __init__(self, root, train, transform=None):

    self.root = root
    self.train = train
    self.transforms = transform
    self._load_data()
    self.image_folder = 'CUB_200_2011/images'

  def _load_data(self):

    # get the path of all the images
    images = pd.read_csv(os.path.join(self.root , 'CUB_200_2011' , 'images.txt'), sep=' ',
                          names=['img_id','path'])
    # get the image class labels
    labels = pd.read_csv(os.path.join(self.root , 'CUB_200_2011' , 'image_class_labels.txt'), sep=' ',
                          names=['img_id','label'])
    # get the training and testing split
    split = pd.read_csv(os.path.join(self.root , 'CUB_200_2011' , 'train_test_split.txt'), sep=' ',
                          names=['img_id','train?'])

    data = images.merge(labels, on='img_id')
    data = data.merge(split, on='img_id')

    if self.train:
      self.data = data[data['train?'] == 1]
    else:
      self.data = data[data['train?'] == 0]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    sample = self.data.iloc[idx]
    path = os.path.join(self.root, self.image_folder, sample.path)
    label = sample.label - 1 # labels start from 1
    img = Image.open(path)
    img = img.convert('RGB')

    if self.transforms is not None:
      img = self.transforms(img)

    return img, label

# trnsformations to be used on the images for training and testing
tnfs = {
    'train': tfs.Compose([
        tfs.Resize([224,224]),
        tfs.RandomRotation(10),
        tfs.RandomHorizontalFlip(),
        tfs.RandomVerticalFlip(),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    'test': tfs.Compose([
        tfs.Resize([224,224]),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
