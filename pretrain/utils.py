"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt 

from torchvision import datasets
from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

def pad_image(image):
    imsize = 84
    full_image = np.zeros((imsize,imsize,3))
    image = np.array(image)

    rand_x = np.random.randint(imsize-image.shape[0])
    rand_y = np.random.randint(imsize-image.shape[1])

    full_image[
        rand_x:rand_x+image.shape[0],
        rand_y:rand_y+image.shape[1], :
    ] = image 

    return full_image


def get_train_valid_loader(data_dir,
                           num_classes,
                           batch_size,
                           val_batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    valid_transform = transforms.Compose([
        # transforms.Lambda(lambda x: pad_image(x)),
        transforms.ToTensor()
    ])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        # transforms.Lambda(lambda x: pad_image(x)),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
  
    # load the dataset
    train_dataset = dataset(
        root=data_dir, 
        train=True,
        download=True, 
        transform=train_transform,
    )

    valid_dataset = dataset(
        root=data_dir, 
        train=True,
        download=True, 
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=val_batch_size, 
        sampler=valid_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    num_classes,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transform
    transform = transforms.Compose([
        # transforms.Lambda(lambda x: pad_image(x)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    dataset = dataset(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=pin_memory,
    )

    return data_loader

if __name__ == "__main__":
    loader = get_test_loader("./data", num_classes=100, batch_size=1)
    image, _ = next(iter(loader))
    transformed_image = pad_image(image[0])
    plt.imshow(np.moveaxis(transformed_image, 0, -1))
    plt.show()