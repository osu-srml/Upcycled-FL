from collections import OrderedDict
from server.avg import Server as ServerBase
import copy
import torch
from client.feddyn import LocalUpdate
from utils import trainable_params,vectorize
import numpy as np

class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.ClientUpdate = LocalUpdate
        
        self.param_numel = vectorize(trainable_params(self.global_model)).numel()
        self.nabla = [
            torch.zeros(self.param_numel, device=self.device) for _ in range(args.num_users)
        ]
    
    def train_one_round(self, model, epoch):
        local_weights = []
        delta_cache = []
        model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = self.ClientUpdate(args=self.args, dataset=self.train_datasets[idx], logger=self.logger)
            w, flag, delta = local_model.update_weights(model=copy.deepcopy(self.global_model), 
                                                                        nabla=self.nabla[idx],)
            local_weights.append((copy.deepcopy(w), self.group_ws[idx]))
            delta_cache.append(delta)
            
            if flag is False:
                continue
        return local_weights, delta_cache
    
        
    def run(self):
        device = self.device
        self.test(0)
        self.set_seed()
        seedcount= 0 
        for epoch in range(self.args.epochs):
            self.global_model.train()
            if (self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==1) or  self.args.algorithm != 'Upcycled':
                np.random.seed(self.seeds[seedcount])
                torch.manual_seed(self.seeds[seedcount])
                seedcount += 1
                prev_model = copy.deepcopy(self.global_model)
                prev_weights = prev_model.state_dict()
                updated_local_weights, delta_cache = self.train_one_round(self.global_model, epoch)
                

                avg_parameters = [
                    torch.stack(params).mean(dim=0) for params in zip(*delta_cache)
                ]
                params_shape = [(param.numel(), param.shape) for param in avg_parameters]
                flatten_avg_parameters = vectorize(avg_parameters)
                
                old_nabla = copy.deepcopy(self.nabla)
                for i, client_params in enumerate(delta_cache):
                    self.nabla[i] += vectorize(client_params) - flatten_avg_parameters

                flatten_new_parameters = flatten_avg_parameters + torch.stack(self.nabla).mean(
                    dim=0
                )
                # reshape
                new_parameters = []
                i = 0
                for numel, shape in params_shape:
                    new_parameters.append(flatten_new_parameters[i : i + numel].reshape(shape))
                    i += numel
                global_params_dict = OrderedDict(
                    zip(self.global_model.state_dict().keys(), new_parameters)
                )
                global_weights = global_params_dict
                
                global_difference = copy.deepcopy(global_weights)
                for key, value in global_weights.items():
                    global_difference[key] = value - prev_weights[key]
                        
                nabla_diff = copy.deepcopy(self.nabla)
                for i, item in enumerate(nabla_diff):
                        item = item - old_nabla[i]
            
            if self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==0 :
                for key, value in global_weights.items():
                    if 'num_batches_tracked' in key:
                        continue
                    global_weights[key] += self.args.upcycled_param * global_difference[key]
                    
                for i, item in enumerate(self.nabla):
                    item = item + self.args.upcycled_param * nabla_diff[i]
            
            self.global_model.load_state_dict(global_weights)          
            self.test(epoch+1)