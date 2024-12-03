import torch

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../'))

import numpy as np
from copy import deepcopy

from worker import Worker
from utils import Server_Trainer
from typing import List
import gc


class Server(Server_Trainer):
    def run_with_multiprocessing(self):
        raise NotImplementedError
    
    def simulation_exp(self, worker_trainers: List[Worker]):
        assert len(self.args.model_size) == len(self.args.model_dist)
        self.args.model_size = list(map(eval, self.args.model_size))
        self.args.model_dist = list(map(eval, self.args.model_dist))
        assert sum(self.args.model_dist) == len(worker_trainers)

        # Create a list of the maximum model size on each client 
        client_model_sizes = []
        for size, n_workers in zip(self.args.model_size, 
                                   self.args.model_dist):
            client_model_sizes.extend([size]*n_workers)

        # Start Training 
        for t in range(self.args.T):
            # Select participants 
            participants = np.random.choice(len(worker_trainers), 
                                            size=self.args.num_part, 
                                            replace=False)
            print('Participants list:', list(participants+1), flush=True)
            
            train_loss = 0.0
            tot_grad, tot_mask = {}, {}
            
            self.model.train()
            
            for name, param in self.model.state_dict().items():
                tot_grad[name] = torch.zeros_like(param)
                tot_mask[name] = torch.zeros_like(param)
            
            for w_idx in participants:
                worker = worker_trainers[w_idx]
                m_size = client_model_sizes[w_idx]
                
                client_model = deepcopy(self.model)
                client_model.layer_wise_generate_mask(model_size=m_size, topk=True, bern=True)
                gradient, loss, iteration = worker.local_training(client_model, m_size)
                
                del client_model
                torch.cuda.empty_cache()
                gc.collect()
                        
                # Aggregate the loss 
                train_loss = train_loss + loss/len(participants)

                # Aggregate the gradient (including normalization 
                # layer, using state_dict())
                for name, grad in gradient.items():
                    tot_grad[name] += grad
                    tot_mask[name] += (grad != 0)
                        
            # Update model
            model_params = self.model.state_dict()
            for name in model_params.keys():
                # Update the parameters 
                if torch.sum(torch.isnan(tot_grad[name])) > 0:
                    print("NaN occurs. Terminate. ", flush=True)
                    return

                # Update with gradient 
                avg_grad = tot_grad[name] / tot_mask[name]
                avg_grad = torch.nan_to_num(avg_grad, nan=0.0, 
                                            posinf=0.0, neginf=0.0)
                if ('running_mean' in name) or ('running_var' in name) \
                    or ('num_batches_tracked' in name):
                    # This is the normalization layer with tracking stats of inputs 
                    model_params[name] = model_params[name] - avg_grad
                else:
                    model_params[name] -= (self.args.lr_global * avg_grad)
            
            self.model.load_state_dict(model_params)
            
            # evaluation 
            if (t+1) % 100 == 0 or (t == 0) or (t >= self.args.T-10):
                eval_client_model_sizes = [1.0, 0.99, 0.9, 0.8, 0.75, 0.64, 
                                           0.7, 0.6, 1./2, 0.36, 1./4, 0.16, 1./8,
                                           1./16, 0.04, 1./32, 1./64, 1./128, 1./256]
                self.eval(eval_client_model_sizes)
            
            if (t+1) % self.args.save_freq == 0 or (t==0) or (t >= self.args.T-100):
                submodel_losses, submodel_accs = [], []
                submodel_local_accs = [0.0]*len(client_model_sizes)

                for m_size in self.args.model_size:
                    test_model = deepcopy(self.model)
                    test_model.layer_wise_generate_mask(m_size, topk=True, bern=True)
                    
                    test_loss, test_acc, correct, total = self.test(test_model)

                    submodel_accs.append(test_acc)
                    submodel_losses.append(test_loss)
                    
                    this_local_accs = []
                    for i, (worker, size) in enumerate(zip(worker_trainers, client_model_sizes)):
                        if size == m_size:
                            _, acc, _, _ = worker.test(test_model)
                            submodel_local_accs[i] = acc
                            this_local_accs.append(acc)
                    print('Round: {}\tModel Size: {:.10f}\tTest Accuracy: {}% ({}/{})\tLocal Accuracy: {}%'\
                            .format(t, m_size, test_acc*100, correct, total, np.average(this_local_accs)*100))
                
                print(f"Round: {t}\tClients' acc: {submodel_local_accs}")
                print('Round: {}\tAverage accuracy: {}%\tLocal accuracy: {}%'.format(
                    t, np.average(submodel_accs)*100, np.average(submodel_local_accs)*100))
                
                sys.stdout.flush()
            
            print(f'Finish the training of Round {t}....')

    def eval(self, client_model_sizes:list):
        # print result (using topk)
        submodel_losses, submodel_accs = [], []

        for m_size in np.unique(client_model_sizes):
            temp_model = deepcopy(self.model)
            temp_model.layer_wise_generate_mask(m_size, topk=True, bern=True)
            test_loss, test_acc, _, _ = self.test(temp_model)
            print('Model Size: {:.10f}\tTest Loss: {}\tTest Accuracy: {}'\
                    .format(m_size, test_loss, test_acc*100))
            submodel_accs.append(test_acc)
            submodel_losses.append(test_loss)
