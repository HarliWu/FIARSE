import os, argparse
import numpy, time

import torch
from torch.utils.data import DataLoader

from typing import List, Callable, Optional, Tuple, Literal
from copy import deepcopy

import sys
sys.setrecursionlimit(10000)


class SGD(torch.optim.SGD):
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[torch.Tensor],
        d_p_list: List[torch.Tensor],
        momentum_buffer_list: List[Optional[torch.Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            cached_param = torch.where(d_p != 0., param, torch.zeros_like(param))
            d_p = d_p.add(cached_param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


# Create the model tester 
def check_accuracy(loader:DataLoader, model:torch.nn.Module, device:torch.device):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # print(targets)
            # print(targets.shape)
            # print(outputs)
            # print(outputs.shape)
            
            # if len(targets.shape) == 2:
            #     targets = targets[:, -1]
            #     outputs = torch.transpose(outputs, 1, 2)
            #     outputs = outputs[:, -1]

            loss = criterion(outputs, targets)
            
            test_loss += loss.item() if len(targets.shape) != 2 else torch.exp(loss).item()
            _, predicted = outputs.max(1)
            total += numpy.prod(targets.size())
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx + 1), correct/total, correct, total


# Create the trainer for the server and the workers 
class Trainer:
    def __init__(self, rank: int, args: argparse.Namespace, model: torch.nn.Module=None, 
                    train_data_loader: DataLoader=None, test_data_loader: DataLoader=None, \
                    multiprocessing: bool=True, cpu: torch.device=None, gpu: torch.device=None):
        self.rank = rank
        self.args = args
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.cpu, self.gpu = cpu, gpu
        self.multiprocessing = multiprocessing

        if self.model is not None:
            self.model = self.model.to(self.gpu)

    def run_with_multiprocessing(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError

    def test(self, model: Optional[torch.nn.Module]=None):
        if self.test_data_loader is None:
            raise ValueError()
        if model is None:
            model = self.model
        return check_accuracy(self.test_data_loader, model, \
                                self.gpu if self.gpu is not None else self.cpu)


class Worker_Trainer(Trainer):
    def __init__(self, rank: int, args: argparse.Namespace, model: torch.nn.Module=None, 
                    train_data_loader: DataLoader=None, test_data_loader: DataLoader=None, \
                    multiprocessing: bool=True, cpu: torch.device=None, gpu: torch.device=None):
        assert rank > 0
        
        if multiprocessing is False:
            model = None
        super().__init__(rank, args, model, train_data_loader, test_data_loader, \
                            multiprocessing, cpu, gpu)

    def get_next_batch(self):
        if not hasattr(self, 'train_iterator'):
            self.train_iterator = iter(self.train_data_loader)
        try:
            return next(self.train_iterator)
        except:
            self.train_iterator = iter(self.train_data_loader)
            return next(self.train_iterator)

    def get_next_K_batch(self, iterations: int=-1, epochs: int=-1, total_size: int=-1):
        training_seq = []
        if iterations > 0:
            for _ in range(iterations):
                training_seq.append(self.get_next_batch())
        elif epochs > 0:
            for _ in range(epochs):
                for tup in self.train_data_loader:
                    training_seq.append(tup)
        elif total_size > 0:
            while total_size > 0:
                training_seq.append(self.get_next_batch())
                total_size -= len(training_seq[-1][0])
        return training_seq

    def get_gradient(self, model: torch.nn.Module, iterations: int=-1, epochs: int=-1, \
                        total_size: int=-1, loss_fn: Callable=torch.nn.CrossEntropyLoss, \
                        reduction: Literal['mean', 'sum']='mean', return_type: type=list):
        # Calculate the gradient on a large set 
        # reduction: 'mean' or 'sum'
        # return: the gradient of trainable parameters 
        model_orig = deepcopy(model.state_dict())
        criterion = loss_fn(reduction=reduction)
        model.zero_grad()
        tot_loops = 0
        for (inputs, targets) in self.get_next_K_batch(iterations=iterations, epochs=epochs, \
                                                            total_size=total_size):
            inputs, targets = inputs.to(self.gpu), targets.to(self.gpu)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            tot_loops = tot_loops + 1

        if return_type is list:
            if reduction == 'mean':
                delta_ws = [param.grad.data.clone().detach()/tot_loops \
                                for param in model.parameters()]
            elif reduction == 'sum':
                delta_ws = [param.grad.data.clone().detach() for param in model.parameters()]
        elif return_type is dict:
            delta_ws = {}
            for name, param in model.state_dict().items():
                if ('running_mean' in name) or ('running_var' in name) \
                    or ('num_batches_tracked' in name):
                    delta_ws[name] = model_orig[name] - param
                else:
                    delta_ws[name] = param.grad.data.clone().detach()/tot_loops if reduction == 'mean' \
                                        else param.grad.data.clone().detach()

        return delta_ws

    # Return loss and gradients 
    def local_training(self, model:torch.nn.Module):
        raise NotImplementedError()

    def start(self, worker_size:int, backend:str='gloo', ip:Tuple[str, str]=('127.0.0.1', '29500'), *args, **kwds):
        print(f'Worker {self.rank} start...', flush=True)

        if self.multiprocessing:
            self.num_workers  = worker_size 
            self.communicator = worker_communicator(rank=self.rank, size=worker_size+1, backend=backend, ip=ip)
            self.run_with_multiprocessing(*args, **kwds)
        else:
            raise NotImplementedError('Multiprocessing is disable. Call local_training(model) instead.')

    # Offload model to file 
    def offload_model(self):
        if hasattr(self, 'model') and self.model is not None:
            self.model_path = './model/{}{}.pt'.format(self.rank, int(numpy.random.rand()*time.time()))
            os.makedirs('./model/', exist_ok=True)
            torch.save(self.model, self.model_path)
            delattr(self, 'model')
            torch.cuda.empty_cache()

    def load_model(self):
        if hasattr(self, 'model'):
            return 
        if hasattr(self, 'model_path'):
            self.model = torch.load(self.model_path)
            self.model = self.model.to(self.gpu)
            os.remove(self.model_path)
            delattr(self, 'model_path')


class Server_Trainer(Trainer):
    def __init__(self, num_workers: int, num_part: int, args: argparse.Namespace, model: torch.nn.Module, 
                    train_data_loader: DataLoader=None, test_data_loader: DataLoader=None, \
                    multiprocessing: bool=True, cpu: torch.device=None, gpu: torch.device=None):
        super().__init__(0, args, model, train_data_loader, test_data_loader, multiprocessing, cpu, gpu)
        self.num_workers = num_workers
        self.num_part = num_part
        print(args)
        
    def terminate(self):
        if self.multiprocessing:
            # Return NaN model 
            NaN_like_model = [torch.ones_like(param)*float('nan') for param in self.model.state_dict().values()]
            self.communicator.send(NaN_like_model, numpy.arange(self.num_workers)+1)

    def simulation_exp(self, worker_trainers: List[Worker_Trainer]):
        raise NotImplementedError()

    def start(self, backend:str='gloo', ip:Tuple[str, str]=('127.0.0.1', '29500'),  *args, **kwds):
        print('Server start...', flush=True)

        if self.multiprocessing:
            self.communicator = server_communicator(size=self.num_workers+1, backend=backend, ip=ip)
            self.run_with_multiprocessing(*args, **kwds)
        else:
            self.simulation_exp(*args, **kwds)

    def early_stopping(self, test_loss:float, test_acc:float, patience:Optional[int]=10):
        if not hasattr(self, 'hist_acc'):
            self.hist_acc, self.hist_loss = [], []
            self.early_stop_patience = patience
            self.early_stop_counter  = 0

        self.hist_acc.append(test_acc)
        self.hist_loss.append(test_loss)

        return False

        if (test_loss > min(self.hist_loss)) and (test_acc < max(self.hist_acc)):
            self.early_stop_counter += 1
            return (self.early_stop_counter >= self.early_stop_patience)
        else:
            self.early_stop_counter = 0

        return False


# Create communicator 
class server_communicator:
    import torch.distributed as dist

    def __init__(self, size: int, backend: str='gloo', device=torch.device('cpu'), \
                    ip:Tuple[str, str]=('127.0.0.1', '29500')):
        self.device = device
        
        if backend == 'mpi':
            self.dist.init_process_group(backend)
        elif backend == 'gloo':
            os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = ip
            self.dist.init_process_group(backend, rank=0, world_size=size)
            # self.dist.init_process_group(backend, rank=0, world_size=size, \
            #                              init_method="file:///d:/test/communicate")  # windows OS

    def send(self, tensor_list: List[torch.Tensor], dst_list: List[int]):
        for dst in dst_list:
            for idx, tensor in enumerate(tensor_list):
                self.dist.send(tensor.clone().detach().to(self.device), dst)

    def recv(self, tensor_list_template: List[torch.Tensor], src_list: List[int]):
        recv_list = []      # aggregate the lists from all senders 
        for src in src_list:
            worker_tensor_list = []
            for idx, _ in enumerate(tensor_list_template):
                tensor = torch.zeros_like(tensor_list_template[idx], device=self.device)
                self.dist.recv(tensor, src)
                worker_tensor_list.append(tensor)
            recv_list.append(worker_tensor_list)
        return recv_list


class worker_communicator:
    import torch.distributed as dist

    def __init__(self, rank: int, size: int, backend='gloo', device=torch.device('cpu'), \
                    ip:Tuple[str, str]=('127.0.0.1', '29500')):
        self.device = device
        
        if backend == 'mpi':
            self.dist.init_process_group(backend)
        elif backend == 'gloo':
            os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = ip
            self.dist.init_process_group(backend, rank=rank, world_size=size)
            # self.dist.init_process_group(backend, rank=rank, world_size=size, \
            #                              init_method="file:///d:/test/communicate")  # windows OS

    def send(self, tensor_list: List[torch.Tensor]):
        for idx, tensor in enumerate(tensor_list):
            self.dist.send(tensor.clone().detach().to(self.device), dst=0)

    def recv(self, tensor_list_template: List[torch.Tensor]):
        server_tensor_list = []     # receive the list from the server 
        for idx, _ in enumerate(tensor_list_template):
            tensor = torch.zeros_like(tensor_list_template[idx], device=self.device)
            self.dist.recv(tensor, src=0)
            server_tensor_list.append(tensor)
        return server_tensor_list
