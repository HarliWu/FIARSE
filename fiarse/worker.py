from copy import deepcopy

import os
import numpy as np
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../'))

from utils import Worker_Trainer
import torch


class Worker(Worker_Trainer):
    def __init__(self, rank, args, model=None, train_data_loader=None, 
                 test_data_loader=None, multiprocessing=True, cpu=None, 
                 gpu=None):
        super().__init__(rank, args, model, train_data_loader, 
                         test_data_loader, multiprocessing, cpu, gpu)

        self.lr = 10**self.args.lr
        self.lr_mask = 10**self.args.lr_mask
        self.local_updates_args = {self.args.K_unit: self.args.K}

    def run_with_multiprocessing(self):
        raise NotImplementedError

    def local_training(self, model:torch.nn.Module, model_size:float):
        # Store the initial information 
        cache_model = deepcopy(model.state_dict())
        client_model = model
        
        # start training 
        criterion = torch.nn.CrossEntropyLoss()
        tot_loss  = 0.0

        client_model.train()

        optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
        
        for iteration, (inputs, targets) in \
            enumerate(self.get_next_K_batch(**self.local_updates_args)):
            inputs, targets = inputs.to(self.gpu), targets.to(self.gpu)
            optimizer.zero_grad()
            
            outputs = client_model(inputs)
            loss = criterion(outputs, targets)
            tot_loss = tot_loss + loss.data
            loss.backward(retain_graph=True)
            optimizer.step()
            
        iteration = iteration + 1
        # Calculate the gradients (including normalization layer) 
        # using state_dict  
        gradient = {}
        for (name, cur_param), (_, cache_param) in \
            zip(client_model.state_dict().items(), cache_model.items()):
            gradient[name] = cache_param - cur_param
        
        print(f'Worker: {self.rank}\tIterations: {iteration}\t'
              f'Loss: {tot_loss/iteration}\tModel size: {model_size}')
        sys.stdout.flush()
        
        return gradient, tot_loss/iteration, iteration
