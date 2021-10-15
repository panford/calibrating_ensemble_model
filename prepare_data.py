import os

import torch
from torchvision import datasets, transforms


def MNIST(data_path ="./data"):
    input_size = 28
  
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    input_dim = train_dataset.data.shape[1]
    num_classes = len(train_dataset.classes)
    return train_dataset, test_dataset, input_dim, num_classes