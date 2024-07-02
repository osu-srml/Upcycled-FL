from server.avg import Server as ServerBase
from tqdm import tqdm
import copy
import torch
from client.avgm import LocalUpdate
import numpy as np

class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.ClientUpdate = LocalUpdate
        
    def run(self):
        device = self.device
        self.global_model.to(device)
        self.test(0)
        self.set_seed()
        seedcount= 0 
        global_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=1.0,
                momentum=self.args.server_momentum,
                nesterov=True,
            )
        for epoch in range(self.args.epochs):
            self.global_model.train()
            if (self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==1) or  self.args.algorithm != 'Upcycled':
                np.random.seed(self.seeds[seedcount])
                torch.manual_seed(self.seeds[seedcount])
                seedcount += 1
                prev_model = copy.deepcopy(self.global_model)
                prev_weights = prev_model.state_dict()
                updated_local_weights = self.train_one_round(self.global_model, epoch)
                
                local_weights, weight_cache =[], []
                delta_list = []
                for l_w, g_w in updated_local_weights:
                    local_weights.append(l_w)
                    weight_cache.append(g_w)
                    
                delta_list = []
                
                for w in local_weights:
                    delta = {}
                    for k in prev_weights.keys():
                        delta[k] =  prev_weights[k] - w[k]
                    delta_list.append(delta)
                
                client_weights = torch.tensor(weight_cache, device=device) / sum(weight_cache)
                aggregated_delta = []
                for k in prev_weights.keys():
                    stacked_deltas = torch.stack([delta[k] for delta in delta_list], dim=-1)
                    aggregated_delta_layer = torch.sum(stacked_deltas * client_weights, dim=-1)
                    aggregated_delta.append(aggregated_delta_layer)

                global_optimizer.zero_grad()
                for param, diff in zip(self.global_model.parameters(), aggregated_delta):
                    param.grad = diff.data
                global_optimizer.step()
                agg_weight = self.global_model.state_dict()
                
                
                global_difference = copy.deepcopy(prev_weights)
                for key, value in agg_weight.items():
                    global_difference[key] = value - prev_weights[key]
            
            if self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==0 :
                for key, value in agg_weight.items():
                    if 'num_batches_tracked' in key:
                        continue
                    agg_weight[key] += self.args.upcycled_param * global_difference[key]
                    
            self.global_model.load_state_dict(agg_weight)
            self.test(epoch+1)