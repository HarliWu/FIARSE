import transformers
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url

from dataclasses import dataclass
from argparse import ArgumentParser
from enum import Enum

import csv

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../'))
from data_utils import BalancedDirichlet, Dirichlet, Pathological, IID, DataPartitioner

# import datasets
# from datasets import load_dataset

_TRAIN_DOWNLOAD_URL = ("https://raw.githubusercontent.com/mhjabreel/"
                       "CharCnn_Keras/master/data/ag_news_csv/train.csv")
_TEST_DOWNLOAD_URL = ("https://raw.githubusercontent.com/mhjabreel/"
                      "CharCnn_Keras/master/data/ag_news_csv/test.csv")

class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.tensor(labels)
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     labels,
        #     batch_first=True,
        #     padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class AGNews(Dataset):
    """
    Load AGNews 
    """
    def __init__(self, root:str, tokenizer=None, train: bool=True, 
                 download: bool=True):
        print(root)
        if download:
            if os.path.exists(os.path.join(root, os.path.basename(_TRAIN_DOWNLOAD_URL))):
                print('Files already downloaded and verified')
            else:
                download_url(_TRAIN_DOWNLOAD_URL, root)
                download_url(_TEST_DOWNLOAD_URL, root)

        if train:
            self.dataset = self._generate_examples(
                os.path.join(root, os.path.basename(_TRAIN_DOWNLOAD_URL)))
        else:
            self.dataset = self._generate_examples(
                os.path.join(root, os.path.basename(_TEST_DOWNLOAD_URL)))
        
        self.data, self.targets = self.dataset['text'], self.dataset['label']
        # self.input_ids = self._tokenize_fn(self.data, tokenizer)['input_ids']
        # for (data, input_ids) in zip(self.data[:100], self.input_ids[:100]):
        #     print(data)
        #     print(input_ids)
        #     print('=======================')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def _generate_examples(self, filepath):
        dataset = {
            "text": [], "label": []
        }
        """Generate AG News examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', 
                delimiter=",", 
                quoting=csv.QUOTE_ALL, 
                skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                label, title, description = row
                # Original labels are [1, 2, 3, 4] ->
                #                   ['World', 'Sports', 'Business', 'Sci/Tech']
                # Re-map to [0, 1, 2, 3].
                label = int(label) - 1
                text = " ".join((title, description))
                # yield id_, {"text": text, "label": label}
                dataset["text"].append(text)
                dataset["label"].append(label)
        return dataset


class LanguageDataset(Dataset):
    def __init__(self, 
                 data: list, 
                 targets: list, 
                 load_data_from='rawdata', 
                 tokenizer=AutoTokenizer.from_pretrained('roberta-base')):
        assert len(data) == len(targets)
        self.data, self.targets = data, targets
        self.input_ids = self._tokenize_fn(self.data, tokenizer)['input_ids']

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.targets[i])

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )


class ServerLoader:
    def __init__(self, parser: ArgumentParser, partitioner: str, 
                 workers: list, dataset_root: str = '~/dataset/',
                 tokenizer=AutoTokenizer.from_pretrained('roberta-base')):
        root = os.path.join(dataset_root.replace('~', os.environ['HOME']), 'agnews_data')
        
        if type(partitioner) is str:
            known_partitioner = {'balanced': BalancedDirichlet, 
                                 'dirichlet': Dirichlet, 
                                 'pathological': Pathological, 
                                 'iid': IID}
            self.partitioner = known_partitioner[partitioner.lower()](parser, len(workers))
        elif type(partitioner) is DataPartitioner:
            self.partitioner = partitioner(parser, len(workers))
        else:
            raise TypeError
        
        # load dataset from a pre-split json file 
        # If not exist, split it now 
        tags_datasets = {
            'train': AGNews(root=root, train=True, tokenizer=tokenizer),
            'test': AGNews(root=root, train=False, tokenizer=tokenizer)
        }
        for tag, dataset in tags_datasets.items():
            print(tag, len(dataset))
            self.partitioner.prepare_dataset(dataset, root, tag)
        
        self.workers = workers
        self.root = root
        self.tokenizer = tokenizer

    def get_loader(self, tag: str, batch_size: int):
        datasets = self.partitioner.load_data(
            self.workers, self.workers, self.root, tag, concat=True,
            DatasetBuilder=LanguageDataset, tokenizer=self.tokenizer)
        return DataLoader(datasets[0], batch_size, shuffle=False, 
                          collate_fn=LLMDataCollator(tokenizer=self.tokenizer))


class ClientLoader:
    def __init__(self, parser: ArgumentParser, partitioner: str, 
                 ranks: list, workers: list, tags: list, 
                 dataset_root: str = '~/dataset/',
                 tokenizer=AutoTokenizer.from_pretrained('roberta-base')):
        root = os.path.join(dataset_root.replace('~', os.environ['HOME']), 'agnews_data')
        
        if type(partitioner) is str:
            known_partitioner = {'balanced': BalancedDirichlet, 
                                 'dirichlet': Dirichlet, 
                                 'pathological': Pathological, 
                                 'iid': IID}
            self.partitioner = known_partitioner[partitioner.lower()](parser, len(workers))
        elif type(partitioner) is DataPartitioner:
            self.partitioner = partitioner(parser, len(workers))
        else:
            raise TypeError

        self.tokenizer = tokenizer
        self.ranks = list(ranks)
        self.datasets = {}
        for tag in tags:
            self.datasets[tag] = self.partitioner.load_data(
                self.ranks, workers, root, tag, 
                DatasetBuilder=LanguageDataset, tokenizer=self.tokenizer)
            
    def get_loader(self, rank, tag:str, batch_size: int):
        assert rank in self.ranks
        print(rank, tag, len(self.datasets[tag][self.ranks.index(rank)]))
        return DataLoader(self.datasets[tag][self.ranks.index(rank)], 
                          batch_size, shuffle=False, 
                          collate_fn=LLMDataCollator(tokenizer=self.tokenizer))
        

if __name__ == "__main__":
    agnews = AGNews('/local/scratch/b/wu1977/dataset')
    # import numpy as np
    
    # parser = ArgumentParser()
    # parser.add_argument('--root', type=str, default='/local/scratch/d/wu1977/dataset/', help='The root of dataset')
    # parser.add_argument('--partitioner', type=str, default='pathological', help='How to partition the dataset')
    # root = parser.parse_known_args()[0].root
    # partitioner = parser.parse_known_args()[0].partitioner

    # workers = np.arange(20)
    # server_data = ServerLoader(parser, partitioner, workers, root)
    # server_data.get_loader('test', 25)
    