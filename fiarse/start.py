# -*- coding: utf-8 -*-
import os, numpy, random, time
import torch

# Create the parser 
import argparse

# import server, worker, and add_parser_arguments
import server as trainer_server
import worker as trainer_worker
import add_parser_arguments


def new_arguments(parser):
    # Method description
    parser.add_argument('--method', type=str, default='FIARSE', help='Running algorithm')

    # Dataset 
    parser.add_argument('--root', type=str, default='~/dataset/', help='The root of dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='The name of dataset used')
    parser.add_argument('--partitioner', type=str, default='dirichlet', help='How to partition the dataset')

    # Model 
    parser.add_argument('--model', type=str, default='BetaResNet18_sbn', help='The name of model used') 

    # Other settings
    parser.add_argument('--bsz', type=int, default=32, help='Batch size for training dataset')
    parser.add_argument('--num-part', type=int, default=10, help='Number of partipants')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomization')
    parser.add_argument('--gpu-idx', action='extend', nargs='+', help='Index of GPU')

    parser.add_argument('--num-workers',type=int, default=100, help='Total number of workers')

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    args = new_arguments(parser)
    method = args.method

    import importlib
    add_parser_arguments.new_arguements(parser)
    # reload parser 
    args = parser.parse_known_args()[0]
    
    # set random seed 
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # import dataset and model 
    import sys
    sys.path.insert(1, '../')
    dataset = importlib.import_module('data.{}.data'.format(args.dataset))
    model = importlib.import_module('model.{}.model'.format(args.dataset))
    # get model
    model = getattr(model, args.model)()

    # Run with pseudo distribution 
    worker_trainers = []
    workers = numpy.arange(args.num_workers) + 1

    # Load Dataset 
    dataset_server = dataset.ServerLoader(
        parser=parser, 
        partitioner=args.partitioner, 
        workers=workers, 
        dataset_root=args.root)
    dataset_client = dataset.ClientLoader(
        parser=parser, 
        partitioner=args.partitioner, 
        ranks=workers, workers=workers, 
        tags=['train', 'test'], 
        dataset_root=args.root)
    
    for idex in range(args.num_workers+1):            
        # Define CPU and GPU
        cpu = torch.device('cpu')
        gpu = torch.device('cuda:{}'.format(args.gpu_idx[0])) \
                    if torch.cuda.is_available() else torch.device('cpu')
        
        if idex == 0:       # This is server 
            # Get Dataloader  
            test_data_loader = dataset_server.get_loader(tag='test', batch_size=args.bsz)

            # Launch Trainer 
            server = trainer_server.Server(num_workers=args.num_workers, 
                                            num_part=args.num_part, 
                                            args=args, model=model, 
                                            train_data_loader=None, 
                                            test_data_loader=test_data_loader, 
                                            multiprocessing=False, cpu=cpu, gpu=gpu)
        
        else:               # This is worker 
            # Get Dataloader
            train_data_loader = dataset_client.get_loader(rank=idex, tag='train', 
                                                            batch_size=args.bsz)
            test_data_loader = dataset_client.get_loader(rank=idex, tag='test', 
                                                            batch_size=args.bsz)

            # Launch Trainer 
            worker = trainer_worker.Worker(rank=idex, args=args, model=model, 
                                            train_data_loader=train_data_loader, 
                                            test_data_loader=test_data_loader, 
                                            multiprocessing=False, 
                                            cpu=cpu, gpu=gpu)
            worker_trainers.append(worker)
            
    server.start(worker_trainers=worker_trainers)

    # Usage: Different approaches should create dedicated server and workers 
    print('\n\n\n\n\n\n\n\n')