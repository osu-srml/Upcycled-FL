from server.avg import Server as ServerBase
from tqdm import tqdm
import copy
import torch
from client.fedyogi import LocalUpdate
from utils import get_dataset, average_weights, exp_details, trainable_params
import numpy as np

# https://arxiv.org/pdf/2003.00295
class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.ClientUpdate = LocalUpdate
        
        self.delta_t_prv = None
        self.v_prv = None
        self.m_t=None

        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.tau = args.tau
        self.parameter_names = [name for name,para in self.global_model.named_parameters()]
        
    def train_one_round(self, model, epoch):
        local_weights = []
        model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
        
        total_samples = 0
        p_i_arr = dict()
        delta = dict()
        for idx in idxs_users:
            local_model = self.ClientUpdate(args=self.args, dataset=self.train_datasets[idx], logger=self.logger)
            w, flag,sd_model_delta = local_model.update_weights(model=copy.deepcopy(model))
            local_weights.append((copy.deepcopy(w), self.group_ws[idx]))
            total_samples += len(self.train_datasets[idx])
            delta[idx] = sd_model_delta
            if flag is False:
                continue
        for idx in idxs_users:
            n_i = len(self.train_datasets[idx])
            p_i_arr[idx] = n_i / total_samples
        return local_weights, p_i_arr, idxs_users, delta
    
    def average_weights(ls):
        sum_ws = 0
        w_avg = copy.deepcopy(ls[0][0])
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                continue
            w_avg[key] *= ls[0][1]
        sum_ws += ls[0][1]
        for i in range(1, len(ls)):
            for key in w_avg.keys():
                if 'num_batches_tracked' in key:
                    continue
                w_avg[key] += (ls[i][0][key] * ls[i][1])
            sum_ws += ls[i][1]
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                continue
            w_avg[key] /= sum_ws
        return w_avg
    
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
                updated_local_weights, p_i_arr, idxs_users, delta = self.train_one_round(self.global_model, epoch)
                
                
                delta_t_cur = copy.deepcopy(prev_weights)
                for name in self.parameter_names:
                    delta_t_cur[name].zero_()
                    
                if self.m_t is None:
                    self.m_t = copy.deepcopy(delta_t_cur)
                
                for client_id in idxs_users:
                    weight_i = p_i_arr[client_id]
                    sd_local_delta = delta[client_id]
                    for name in self.parameter_names:
                        delta_t_cur[name] += sd_local_delta[name] * weight_i
                
                # Paper Algorithm 2
                for name in self.parameter_names:
                    self.m_t[name] = self.beta1 * self.m_t[name] + (1 - self.beta1) * delta_t_cur[name]
                    
                if self.v_prv is None:
                    self.v_prv = copy.deepcopy(prev_weights)
                    for name in self.parameter_names:
                        self.v_prv[name].zero_()

                v_t = copy.deepcopy(self.v_prv)
                for name in self.parameter_names:
                    v_t[name] = self.v_prv[name] \
                                - (1 - self.beta2) * delta_t_cur[name] * delta_t_cur[name] \
                                * torch.sign(self.v_prv[name] - delta_t_cur[name] * delta_t_cur[name])
                self.v_prv = v_t

                agg_weight = copy.deepcopy(prev_weights)
                for name in self.parameter_names:
                    agg_weight[name] = prev_weights[name] + self.args.yogi_lr * self.m_t[name]/(torch.sqrt(v_t[name]) + self.tau)
                    
                global_difference = copy.deepcopy(agg_weight)
                for key, value in agg_weight.items():
                    global_difference[key] = value - prev_weights[key]
            

            
            if self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==0 :
                for key, value in agg_weight.items():
                    if 'num_batches_tracked' in key:
                        continue
                    agg_weight[key] += self.args.upcycled_param * global_difference[key]
            self.global_model.load_state_dict(agg_weight)
            self.test(epoch+1)