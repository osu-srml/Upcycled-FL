from server.avg import Server as ServerBase
from tqdm import tqdm
import copy
import torch
from client.scaffold import LocalUpdate
from utils import  trainable_params
import numpy as np

class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.ClientUpdate = LocalUpdate
        
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.global_model)
        ]
        
        self.c_locals = {}
        for client in range(args.num_users):
            self.c_locals[client] = [torch.zeros_like(c) for c in self.c_global]
    
    def train_one_round(self, model, epoch):
        local_weights = []
        c_delta_cache = []
        model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = self.ClientUpdate(args=self.args, dataset=self.train_datasets[idx], logger=self.logger)
            w, flag, c_local, c_delta = local_model.update_weights(model=copy.deepcopy(model), c_global = self.c_global,c_local=self.c_locals[idx])
            local_weights.append((copy.deepcopy(w), self.group_ws[idx]))

            self.c_locals[idx] = c_local
            c_delta_cache.append(c_delta)
            
        return local_weights, c_delta_cache
    
    def run(self):
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
                updated_local_weights, c_delta_cache = self.train_one_round(self.global_model, epoch)
                
                agg_weight, global_difference = self.agg(updated_local_weights, prev_weights)
                
                # update global control
                for c_g_temp, c_delta in zip(self.c_global, zip(*c_delta_cache)):
                    c_delta = torch.stack(c_delta, dim=-1).sum(dim=-1)
                    c_g_temp.data += (1 / self.args.num_users) * c_delta.data

            
            if self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==0 :
                for key, value in agg_weight.items():
                    if 'num_batches_tracked' in key:
                        continue
                    agg_weight[key] += self.args.upcycled_param * global_difference[key]
                    
                # update global control
                for c_g_temp, c_delta in zip(self.c_global, zip(*c_delta_cache)):
                    c_delta = torch.stack(c_delta, dim=-1).sum(dim=-1)
                    c_g_temp.data += self.args.upcycled_param * (1 / self.args.num_users) * c_delta.data  
            self.global_model.load_state_dict(agg_weight)
            self.test(epoch+1)