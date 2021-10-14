import os

import torch
# from torch.utils.data import Dataset

from torchvision import datasets, transforms

# from scipy.io import loadmat
# from PIL import Image

def MNIST(root="./"):
    input_size = 28
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root + "data/", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root + "data/", train=False, download=True, transform=transform
    )
    return input_size, num_classes, train_dataset, test_dataset