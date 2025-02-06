import json
import itertools
import re
import string
import unicodedata
import random

import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from torchtext.datasets import AG_NEWS, IMDB, SogouNews
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


nlp_datasets = ["agnews", "imdb", "sogou"]

import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y


def get_dataset(args):
    transform = None
    if args.dataset == 'mnist':
        data = datasets.MNIST(root=".", download=True)
        n_classes = 10
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == 'fmnist':
        data = datasets.FashionMNIST(root=".", download=True)
        n_classes = 10
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == 'emnist':
        data = datasets.EMNIST(root=".", split="balanced", download=True)
        n_classes = 62
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == 'cifar10':
        data = datasets.CIFAR10(root=".", download=True)
        n_classes = 10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif args.dataset == 'cifar100':
        data = datasets.CIFAR100(root=".", download=True)
        n_classes = 100
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    elif args.dataset == "agnews":
        tokenizer = get_tokenizer("basic_english")
        data = []
        train_data, test_data = list(AG_NEWS(root=".", split="train")), list(AG_NEWS(root=".", split="test"))
        data.extend(train_data)
        data.extend(test_data)
        n_classes = 4

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)


        vocab = build_vocab_from_iterator(yield_tokens(iter(data)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
    elif args.dataset == 'imdb':
        tokenizer = get_tokenizer("basic_english")
        data = []
        train_data, test_data = list(IMDB(root=".", split="train")), list(IMDB(root=".", split="test"))
        data.extend(train_data)
        data.extend(test_data)
        n_classes = 2

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)


        vocab = build_vocab_from_iterator(yield_tokens(iter(data)), specials=["<unk>"], max_tokens=131072)
        vocab.set_default_index(vocab["<unk>"])
    elif args.dataset == 'sogou':
        tokenizer = get_tokenizer("basic_english")
        data = []
        train_data, test_data = list(SogouNews(root=".", split="train")), list(SogouNews(root=".", split="test"))
        data.extend(train_data)
        data.extend(test_data)
        n_classes = 5

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)


        vocab = build_vocab_from_iterator(yield_tokens(iter(data)), specials=["<unk>"], max_tokens=131072)
        vocab.set_default_index(vocab["<unk>"])
    else:
        raise NotImplementedError

    idcs = np.random.permutation(len(data))
    if args.dataset == 'emnist':
        train_idcs, test_idcs = idcs[:50000], idcs[50000:60000]
    else:
        split = int(len(data) * args.train_frac)
        train_idcs, test_idcs = idcs[:split], idcs[split:]

    if args.dataset in nlp_datasets:
        train_labels, formatted_data = [], []
        for l, t in data:
            token = vocab(tokenizer(t))
            label = l - 1
            formatted_data.append((token, label))
            train_labels.append(label)
        train_labels = np.array(train_labels)
        data = formatted_data
    else:
        train_labels = np.array(data.targets)

    client_idcs = split_noniid(train_idcs, train_labels, alpha=args.dirichlet, n_clients=args.n_client)

    split_labels = [{} for _ in range(args.n_client)]
    for i, idcs in enumerate(client_idcs):
        for label in train_labels[idcs]:
            split_labels[i][label] = split_labels[i].get(label, 0) + 1
        print("Client %d: %s" % (i, split_labels[i]))

    client_data = [CustomSubset(data, idcs, transform) for idcs in client_idcs]
    test_data = CustomSubset(data, test_idcs, transform)

    return client_data, test_data, n_classes
