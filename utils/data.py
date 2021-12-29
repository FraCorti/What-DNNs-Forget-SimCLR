import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def get_cifar10(batch_size=128, workers=0, shuffle=False):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    transform_validation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)
    loader_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=workers)

    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)
    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, num_workers=workers)
    loaders = {"train": loader_train, "valid": loader_valid}

    return loaders


def get_ordered_cifar10_validation(batch_size=128, workers=0):
    transform_validation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)
    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, num_workers=workers)
    loaders = OrderedDict()
    loaders["valid"] = loader_valid
    return loaders


def plot_loss_curves(loss_train, loss_valid, final_sparsity, epochs, depth, dropout, model_id):
    iterations = np.arange(0, len(loss_train), 1)

    plt.clf()
    plt.plot(iterations, loss_train, 'b-')
    plt.plot(iterations, loss_valid, 'r-')
    plt.grid(True)
    plt.legend(["Loss training", "Loss validation"], loc="upper right")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.figure()
    plt.savefig(
        '/home/f/fraco1997/compressed_model_v2/result/loss_curves_{}pruning_{}epochs_{}depth_{}dropout_id{}.png'.format(
            final_sparsity, epochs, depth, dropout, model_id))


def plot_accuracy_curves(accuracy_train, accuracy_valid, final_sparsity, epochs, depth, dropout, model_id):
    iterations = np.arange(0, len(accuracy_train), 1)

    plt.clf()
    plt.plot(iterations, accuracy_train, 'b-')
    plt.plot(iterations, accuracy_valid, 'r-')
    plt.grid(True)
    plt.legend(["Accuracy training", "Accuracy validation"], loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.figure()
    plt.savefig(
        '/home/f/fraco1997/compressed_model_v2/result/accuracy_curves_{}pruning_{}epochs_{}depth_{}dropout_id{}.png'.format(
            final_sparsity, epochs, depth, dropout, model_id))
