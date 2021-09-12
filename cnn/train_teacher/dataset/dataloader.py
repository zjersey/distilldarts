import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def dogvcat_train_loader(path, batch_size=32, kwargs={}):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

def dogvcat_test_loader(path, batch_size=32, kwargs={}):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        **kwargs)


def ImageNet_train_loader(path, batch_size=32, kwargs={}):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

def ImageNet_test_loader(path, batch_size=32, kwargs={}):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        **kwargs)

def cifar10_train_loader(path, batch_size=32, kwargs={}):
    return data.DataLoader(
        datasets.CIFAR10(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, 
        shuffle=True,
        **kwargs)

def cifar10_test_loader(path, batch_size=32, kwargs={}):
    return data.DataLoader(
        datasets.CIFAR10(path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, 
        shuffle=True, 
        **kwargs)

def cifar100_train_loader(path, batch_size=32, kwargs={}):
    return data.DataLoader(
        datasets.CIFAR100(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, 
        shuffle=True,
        **kwargs)

def cifar100_test_loader(path, batch_size=32, kwargs={}):
    return data.DataLoader(
        datasets.CIFAR100(path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, 
        shuffle=True, 
        **kwargs)


def trian_dataloader(dataset, path, batch_size, kwargs):
    if dataset=='dog_vs_cat':
        return dogvcat_train_loader(path, batch_size, kwargs)
    elif dataset=='imagenet':
        return ImageNet_train_loader(path, batch_size, kwargs)
    elif dataset=='cifar10':
        return cifar10_train_loader(path, batch_size, kwargs)
    elif dataset=='cifar100':
        return cifar100_train_loader(path, batch_size, kwargs)
    else:
        raise(KeyError("no dataseet named as %s"%dataset))

def test_dataloader(dataset, path, batch_size, kwargs):
    if dataset=='dog_vs_cat':
        return dogvcat_test_loader(path, batch_size, kwargs)
    elif dataset=='imagenet':
        return ImageNet_test_loader(path, batch_size, kwargs)
    elif dataset=='cifar10':
        return cifar10_test_loader(path, batch_size, kwargs)
    elif dataset=='cifar100':
        return cifar100_test_loader(path, batch_size, kwargs)
    else:
        raise(KeyError("no dataseet named as %s"%dataset))

if __name__ == '__main__':
    path = '../data/cifar100'
    train_dataset = datasets.CIFAR100(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    test_dataset = datasets.CIFAR100(path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))

    print(len(train_dataset), len(test_dataset))
