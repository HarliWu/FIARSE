from argparse import ArgumentParser
import numpy as np
import os, shutil
import json
from PIL import Image
import matplotlib.pyplot as plt

from typing import Union, Optional, Callable, List, Tuple, Dict, Literal
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, data: list, 
                 targets: list, 
                 load_data_from: Literal['rawdata', 'path']):
        assert len(data) == len(targets)
        self.data, self.targets = data, targets
        self.load_data_from = load_data_from

    def _get_item_by_rawdata(self, index):
        raise NotImplementedError

    def _get_item_by_path(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.load_data_from == 'rawdata':
            return self._get_item_by_rawdata(index)
        elif self.load_data_from == 'path':
            return self._get_item_by_path(index)
        else:
            raise NotImplementedError

class VisionDataset(myDataset):
    def __init__(self, 
                 data: list, 
                 targets: list, 
                 load_data_from: Literal['rawdata', 'path'], 
                 transform: Optional[Callable]=None, 
                 target_transform: Optional[Callable]=None):
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(data, targets, load_data_from)

    def __len__(self):
        return len(self.data)

    def _get_item_by_rawdata(self, index):
        data, target = np.array(self.data[index]), self.targets[index]

        if len(data.shape) == 2:
            # print('This is a grey image')
            img = Image.fromarray(data.astype('uint8'), mode="L")
        elif len(data.shape) == 3:
            # print('This is an RGB image')
            img = Image.fromarray(data.astype('uint8'), mode="RGB")
        else:
            raise TypeError('Cannot convert the array to images with the size {}'.format(data.shape))
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _get_item_by_path(self, index):
        path, target = self.data[index], self.targets[index]

        with open(path, "rb") as f:     # Load image 
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # Plot the data 
    def plot_data(self, name='xx.png', columns=6, rows=3, pic_indices: Optional[List]=None):
        fig = plt.figure(figsize=(13, 8))
        # ax enables access to manipulate each of subplots
        ax = []

        for i in range(columns * rows):
            if pic_indices is None:
                img, label = self[i]
            elif i >= len(pic_indices):
                break
            else:
                img, label = self[pic_indices[i]]
            # print(np.array(img), label)
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title("Label: " + str(label))  # set title
            plt.imshow(img.T)

        plt.savefig(name)
        # plt.show()  # finally, render the plot

class LanguageDataset(myDataset):
    pass

def name_builder(partition_name: str, num_partitions: int, hyperparameters: dict):
    name = f'{partition_name.lower()}_{num_partitions}parts'
    for key in sorted(hyperparameters.keys()): 
        name = name + f'_{key}{hyperparameters[key]}'
    return name

'''
    Data partitioner: A toolkit 
    :param data: imgs from dataset such as CIFAR10 (Client should set none) 
    :param psizes: int or list, indicating the number of workers 
    i.i.d. partition the dataset into [training, validate, test] 
'''
class DataPartitioner:
    def __init__(self, parser: ArgumentParser, psizes: Union[int, list]):
        self.hyperparameters = self.add_arguments(parser)
        self.psizes = psizes if type(psizes) is list else np.ones(psizes)/psizes
        self.net_cls = None

    def partition_approach(self, dataset: Dataset, psizes: list, net_cls: Optional[list], **kwargs):
        # Return: net_dataidx_map (Dict: {id: indices}), weights
        raise NotImplementedError

    def add_arguments(self, parser: ArgumentParser): 
        return {}
    
    def prepare_dataset(self, dataset: Dataset, root: str, tag: str='train', \
                        split_by_partitions: bool=False, overwrite: bool=False):
        if hasattr(dataset, 'samples'):
            self.prepare_dataset_by_path(dataset, root, tag, split_by_partitions, overwrite)
        else:
            self.prepare_dataset_by_rawdata(dataset, root, tag, split_by_partitions, overwrite)

    def check_exist(self, root: str, name: str, tag: str):
        if os.path.exists(os.path.join(root, name)):
            for dir in os.listdir(os.path.join(root, name)):
                if tag in dir:
                    return True
        return False

    def prepare_dataset_by_rawdata(self, dataset: Dataset, root: str, tag: str='train', \
                                    split_by_partitions: bool=False, overwrite: bool=False):
        # Return: Dict('users': [], 'num_samples': [], 
        #              'user_data': {user_name: {'x': [], 'y': []}})
        name = name_builder(self.__class__.__name__, len(self.psizes), self.hyperparameters)
        if not os.path.exists(os.path.join(root, name)):
            os.makedirs(os.path.join(root, name))
        elif overwrite:
            # Cover the original directory
            shutil.rmtree(os.path.join(root, name))
            os.makedirs(os.path.join(root, name))
        else: 
            # The directory and the file already exist... 
            # Load the file directly 
            if self.check_exist(root, name, tag):
                return 

        net_dataidx_map, weight = self.partition_approach(dataset, self.psizes, \
                                                            self.net_cls, **self.hyperparameters)
        
        net_cls = []
        if split_by_partitions:
            # Each partition for one single file (by "client name + tag")
            for net_i, dataidx in net_dataidx_map.items():
                client_name = 'c{:06d}'.format(net_i)
                records = {
                    'users': [client_name],
                    'num_samples': [len(dataidx)],
                    'user_data': {client_name: {'x': [], 'y': []}}
                }
                net_cls.append([])

                for idx in dataidx:
                    data, target = dataset[idx]
                    records['user_data'][client_name]['x'].append(np.array(data).tolist())
                    records['user_data'][client_name]['y'].append(target)
                    if target not in net_cls[-1]:
                        net_cls[-1].append(target)
                
                with open(os.path.join(root, name, client_name+f'_{tag}.json', 'w')) as f:
                    f.write(json.dumps(records))
            
        else:
            # All partition should put together
            records = {
                'users': [],
                'num_samples': [],
                'user_data': {}
            } 

            for net_i, dataidx in net_dataidx_map.items():
                client_name = 'c{:06d}'.format(net_i)
                records['users'].append(client_name)
                records['num_samples'].append(len(dataidx))
                records['user_data'][client_name] = {'x': [], 'y': []}
                net_cls.append([])

                for idx in dataidx:
                    data, target = dataset[idx]
                    records['user_data'][client_name]['x'].append(np.array(data).tolist())
                    records['user_data'][client_name]['y'].append(target)
                    if target not in net_cls[-1]:
                        net_cls[-1].append(target)

            with open(os.path.join(root, name, f'{tag}.json'), 'w') as f:
                f.write(json.dumps(records))

        if self.net_cls is None:
            self.net_cls = net_cls
        
    def prepare_dataset_by_path(self, dataset: Dataset, root: str, tag: str='train', \
                                split_by_partitions: bool=False, overwrite: bool=False):
        # Return: Dict('users': [], 'num_samples': [], 'user_data': {user_name: {'path': [], 'y': []}})
        name = name_builder(self.__class__.__name__, len(self.psizes), self.hyperparameters)
        if not os.path.exists(os.path.join(root, name)):
            os.makedirs(os.path.join(root, name))
        elif overwrite:
            # Cover the original directory
            shutil.rmtree(os.path.join(root, name))
            os.makedirs(os.path.join(root, name))
        else:
            # The directory and the file already exist... Load the file directly 
            if self.check_exist(root, name, tag):
                return 

        net_dataidx_map, weights = self.partition_approach(dataset, self.psizes, \
                                                            self.net_cls, **self.hyperparameters)
        
        net_cls = []
        if split_by_partitions:
            # Each partition for one single 
            for net_i, dataidx in net_dataidx_map.items():
                client_name = 'c{:06d}'.format(net_i)
                records = {
                    'users': [client_name],
                    'num_samples': [len(dataidx)],
                    'user_data': {client_name: {'path': [], 'y': []}}
                }
                net_cls.append([])

                base_root = os.path.join(root, name, client_name)
                os.makedirs(base_root)
                for idx in dataidx:
                    data, target = dataset[idx]
                    im = Image.fromarray(data)
                    im_path = os.path.join(base_root, '{tag}{:06d}.jpg'.format(idx))
                    im.save(im_path)
                    records['user_data'][client_name]['path'].append(im_path)
                    records['user_data'][client_name]['y'].append(target)
                    if target not in net_cls[-1]:
                        net_cls[-1].append(target)

                with open(os.path.join(root, name, client_name+f'_{tag}.json'), 'w') as f:
                    f.write(json.dumps(records))

        else:
            # All partition should put together
            if hasattr(dataset, 'samples'):
                # make the original path remains unchanged
                records = {
                    'users': [],
                    'num_samples': [],
                    'user_data': {}
                } 

                for net_i, dataidx in net_dataidx_map.items():
                    client_name = 'c{:06d}'.format(net_i)
                    records['users'].append(client_name)
                    records['num_samples'].append(len(dataidx))
                    records['user_data'][client_name] = {'path': [], 'y': []}
                    net_cls.append([])

                    for idx in dataidx:
                        path, target = dataset.samples[idx]
                        records['user_data'][client_name]['path'].append(path)
                        records['user_data'][client_name]['y'].append(target)
                        if target not in net_cls[-1]:
                            net_cls[-1].append(target)

                with open(os.path.join(root, name, f'{tag}.json'), 'w') as f:
                    f.write(json.dumps(records))
            
            else:
                # create a new folder to store 
                # This should put as json file 
                # Please use prepare_dataset_by_rawdata()
                raise NotImplementedError

        if self.net_cls is None:
            self.net_cls = net_cls

    def load_data(self, 
                  ranks: list, 
                  workers: list, 
                  root: str, 
                  tag: str='train', 
                  concat: bool=False, 
                  DatasetBuilder=myDataset, **kwargs):
        assert set(ranks).issubset(set(workers))
        
        name = name_builder(self.__class__.__name__, len(self.psizes), self.hyperparameters)
        assert os.path.exists(os.path.join(root, name))
        records = {
            'users': [],
            'num_samples': [],
            'user_data': {}
        } 

        # Load data from json files 
        for data_file in os.listdir(os.path.join(root, name)):
            if (data_file.endswith('.json')) and (tag in data_file):
                with open(os.path.join(root, name, data_file), 'r') as f:
                    new_records = json.loads(f.read())
                
                for user, n_samples in zip(new_records['users'], new_records['num_samples']):
                    records['users'].append(user)
                    records['num_samples'].append(n_samples)
                    records['user_data'][user] = new_records['user_data'][user]

        # Create dataset for each rank 
        assert len(records['users']) == len(ranks) or len(records['users']) == len(workers)
        datasets = [0] * len(ranks)
        for idx, rank in enumerate(ranks):
            user = records['users'][idx] if len(records['users']) == len(ranks) \
                                        else records['users'][workers.index(rank)]
            user_data = records['user_data'][user]
            if 'path' in user_data:
                datasets[idx] = DatasetBuilder(data=user_data['path'], targets=user_data['y'], \
                                                load_data_from='path', **kwargs)
            elif 'x' in user_data:
                datasets[idx] = DatasetBuilder(data=user_data['x'], targets=user_data['y'], \
                                                load_data_from='rawdata', **kwargs)
            else:
                raise ValueError

        if concat:
            datasets = [sum(datasets[1:], datasets[0])]
            
        return datasets

class Dirichlet(DataPartitioner):
    def partition_approach(self, dataset: Dataset, psizes: list, net_cls: Optional[list], alpha: float):
        n_nets = len(psizes)
        labelList = np.array(dataset.targets)
        K = len(np.unique(labelList))
        min_size = 0
        N = len(labelList)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # net_cls_counts = {}

        # for net_i, dataidx in net_dataidx_map.items():
        #     unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        #     tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        #     net_cls_counts[net_i] = tmp
        # print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        # print(weights)

        return net_dataidx_map, weights

    def add_arguments(self, parser: ArgumentParser):
        if not hasattr(parser.parse_known_args()[0], 'alpha'):
            parser.add_argument('--alpha', type=float, default=0.1, help='The alpha value for dirichlet distrition')
        alpha = parser.parse_known_args()[0].alpha

        return {'alpha': alpha}


class BalancedDirichlet(DataPartitioner):
    def partition_approach(self, dataset: Dataset, psizes: list, net_cls: Optional[list], alpha: float):
        n = len(psizes)
        n_nets = len(psizes)
        labelList = np.array(dataset.targets)
        K = len(np.unique(labelList))

        labelList_true = np.array(dataset.targets)

        min_size = 0
        N = len(labelList_true)
        rnd = 0

        net_dataidx_map = {}

        p_client = np.zeros((n,K))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,K))
            
        p_client_cdf = np.cumsum(p_client, axis=1)
        
        idx_batch = [[] for _ in range(n)]
        
        m = int(N/n)
        
        idx_labels = [np.where(labelList_true==k)[0] for k in range(K)]
        idx_counter = [0 for k in range(K)]
        total_cnt = 0
        
        while(total_cnt<m*n):
            curr_clnt = np.random.randint(n)
            if (len(idx_batch[curr_clnt])>=m):
                continue

            total_cnt += 1
            curr_prior = p_client_cdf[curr_clnt]
                
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if (idx_counter[cls_label] >= len(idx_labels[cls_label])):
                    continue

                idx_batch[curr_clnt].append(idx_labels[cls_label][idx_counter[cls_label]])
                idx_counter[cls_label] += 1

                break

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList_true[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        print(net_cls_counts)

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        return net_dataidx_map, weights

    def add_arguments(self, parser: ArgumentParser):
        if not hasattr(parser.parse_known_args()[0], 'alpha'):
            parser.add_argument('--alpha', type=float, default=0.3, help='The alpha value for dirichlet distrition')
        alpha = parser.parse_known_args()[0].alpha

        return {'alpha': alpha}


class Pathological(DataPartitioner):
    def partition_approach(self, dataset: Dataset, psizes: list, net_cls: Optional[list], classes: int):
        labelList = np.array(dataset.targets)
        K = len(np.unique(labelList))

        if net_cls is None:
            classes = -1 if len(psizes) * classes < K and len(psizes) < K else max(min(classes, K), 1)
            all_labels = np.unique(labelList)
            if classes == -1:
                worker_labels = np.array_split(all_labels, len(psizes))
            else: 
                worker_labels = np.array([], dtype=int)
                for _ in range(int(len(psizes) * classes / K)):
                    worker_labels = np.concatenate([worker_labels, \
                                                    np.random.choice(all_labels, size=K, replace=False)])
                worker_labels = np.array_split(worker_labels, len(psizes))
                
                # worker_labels = np.concatenate([np.random.choice(all_labels, size=K, replace=False), 
                #                                 np.random.choice(all_labels, size=len(psizes)*classes-K, 
                #                                                  replace=True)]).reshape(len(psizes), classes)
                # # in case that some sets exist duplicate 
                # for idx, _ in enumerate(worker_labels):
                #     while len(np.unique(worker_labels[idx])) != classes:
                #         worker_labels[idx] = np.random.choice(all_labels, classes, replace=False)
        else:
            worker_labels = np.array(net_cls)
        
        print('worker_labels:', worker_labels)
        
        data_dict = {}      # storing the iterators for each label 
        for k in np.unique(labelList):
            data_dict[k] = np.where(labelList == k)[0]  # indices of k-th class 
            np.random.shuffle(data_dict[k])
            data_dict[k] = iter(np.array_split(data_dict[k], \
                                                len(np.where(np.concatenate(worker_labels) == k)[0])))
        
        net_dataidx_map = {}

        for idx, _ in enumerate(psizes):
            net_dataidx_map[idx] = []
            for label in worker_labels[idx]:
                net_dataidx_map[idx] += list(next(data_dict[label]))
            np.random.shuffle(net_dataidx_map[idx])
        
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i, _ in enumerate(psizes):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)

        return net_dataidx_map, weights

    def add_arguments(self, parser: ArgumentParser):
        if not hasattr(parser.parse_known_args()[0], 'classes'):
            parser.add_argument('--classes', type=int, default=2, help='Number of classes on each worker')
        classes = parser.parse_known_args()[0].classes

        return {'classes': classes}

class IID(DataPartitioner):
    def partition_approach(self, dataset: Dataset, psizes: list, net_cls: Optional[list] = None, **kwargs):
        self.partitions = {}
        self.ratio = psizes
        data_len = len(dataset) 
        indexes = [x for x in range(0, data_len)] 
        np.random.shuffle(indexes) 
    
        for idx, frac in enumerate(psizes): 
            part_len = int(frac * data_len)
            self.partitions[idx] = indexes[0:part_len]
            indexes = indexes[part_len:]
        
        return self.partitions, psizes

    def add_arguments(self, parser: ArgumentParser):
        return super().add_arguments(parser)

'''
    A template to load the dataset 
    You must implement preload() 
'''     
class _ServerLoader:
    def __init__(self, parser: ArgumentParser, partitioner: str, \
                    workers: list, dataset_root: str = '~/dataset/'):
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
        
        dataset_root = dataset_root.replace('~', os.environ['HOME'])
        tags_datasets, DatasetBuilder, kwargs = self.preload(dataset_root)
        if tags_datasets is not None:
            for tag, dataset in tags_datasets.items():
                print(tag, len(dataset))
                self.partitioner.prepare_dataset(dataset, dataset_root, tag) 
                
        self.workers = workers
        self.DatasetBuilder = DatasetBuilder
        self.kwargs = kwargs
        self.root = dataset_root
        
    def get_loader(self, tag: str, batch_size: int):
        datasets = self.partitioner.load_data(self.workers, self.workers, 
                                              self.root, tag, 
                                              concat=True, 
                                              DatasetBuilder=self.DatasetBuilder, 
                                              **self.kwargs)
        print('server', tag, len(datasets[0]))
        return DataLoader(datasets[0], batch_size, shuffle=False)

    def preload(self, root):
        # Return: tags_datasets, DatasetBuilder, **kwargs 
        raise NotImplementedError

class _ClientLoader:
    def __init__(self, parser: ArgumentParser, partitioner: str, \
                    ranks: list, workers: list, tags: list, dataset_root: str = '~/dataset/'):
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
        
        dataset_root = dataset_root.replace('~', os.environ['HOME'])
        DatasetBuilder, kwargs = self.preload(dataset_root)
        self.ranks = list(ranks)
        self.datasets = {}
        for tag in tags:
            self.datasets[tag] = self.partitioner.load_data(
                self.ranks, workers, dataset_root, tag, 
                DatasetBuilder=DatasetBuilder, **kwargs)
            
    def get_loader(self, rank, tag: str, batch_size: int):
        assert rank in self.ranks
        print(rank, tag, len(self.datasets[tag][self.ranks.index(rank)]))
        return DataLoader(self.datasets[tag][self.ranks.index(rank)], 
                          batch_size, 
                          shuffle=False)
    
    def preload(self, root):
        # Return: DatasetBuilder, **kwargs 
        raise NotImplementedError