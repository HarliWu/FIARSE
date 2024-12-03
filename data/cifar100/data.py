from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

import numpy as np
import os
dirname = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(dirname, '../'))
from data_utils import _ServerLoader, _ClientLoader, VisionDataset

from argparse import ArgumentParser

class ServerLoader(_ServerLoader):
    def preload(self, root):
        tags_datasets = {
            'train': CIFAR100(root=root, train=True, download=True), 
            'test' : CIFAR100(root=root, train=False, download=True)
        }
        transform = transforms.Compose([ 
                        transforms.ToTensor(), 
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
                    ])      # Transform for test dataset
        
        return tags_datasets, VisionDataset, {'transform': transform}
        
    def __init__(self, parser: ArgumentParser, partitioner: str, \
                    workers: list, dataset_root: str = '~/dataset/'):
        root = dataset_root.replace('~', os.environ['HOME']) + '/cifar100_data'
        
        super().__init__(parser=parser, partitioner=partitioner, \
                            workers=workers, dataset_root=root)

class ClientLoader(_ClientLoader):
    def preload(self, root):
        transform = transforms.Compose([ 
                        transforms.RandomCrop(32, padding=4), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor(), 
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
                    ])      # Transform for train dataset
        
        return VisionDataset, {'transform': transform}    
    
    def __init__(self, parser: ArgumentParser, partitioner: str, \
                    ranks: list, workers: list, tags: list, dataset_root: str = '~/dataset/'):
        root = dataset_root.replace('~', os.environ['HOME']) + '/cifar100_data'
        
        super().__init__(parser, partitioner, ranks, workers, tags, root)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='~/dataset/', help='The root of dataset')
    parser.add_argument('--partitioner', type=str, default='pathological', help='How to partition the dataset')
    root = parser.parse_known_args()[0].root
    partitioner = parser.parse_known_args()[0].partitioner

    workers = np.arange(20)
    server_data = ServerLoader(parser, partitioner, workers, root)
    server_data.get_loader('test', 25)