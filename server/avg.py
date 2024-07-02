
import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from utils import get_dataset, average_weights, exp_details
from models import LogisticRegression, Twolayer, SentimentLSTM
from client.avg import LocalUpdate
import time
class Server:
    def __init__(self,args):
        self.logger = SummaryWriter('../logs')
        self.args = args
        if torch.cuda.is_available(): 
            self.device = "cuda" 
        else: 
            self.device = "cpu" 
        print("device", self.device)
        self.train_datasets, self.test_dataset, self.user_groups, self.group_ws = get_dataset(args)
        
        # build model
        if args.dataset == 'femnist' :
            self.global_model = Twolayer(args=args)
            args.feature_len = 28*28
        if "synthetic" in args.dataset:
            self.global_model = LogisticRegression(args=args)
            args.feature_len = 20
        if "sent140" in args.dataset:
            self.global_model = SentimentLSTM()
            args.feature_len = 256
        if  args.dataset == 'nist':
            self.global_model = Twolayer(args=args)
            args.feature_len = 28*28
        self.global_model.to(self.device)
        args.num_users = len(self.user_groups.keys())
        exp_details(args)
        
        self.ClientUpdate = LocalUpdate
        self.train_agents = [self.ClientUpdate(args=self.args, dataset=self.train_datasets[c], logger=self.logger) for c in range(self.args.num_users)]
        self.test_agent = self.ClientUpdate(args=self.args, dataset=self.test_dataset, logger=self.logger)
        
    def set_seed(self):
        seed = self.args.seed
        # random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seeds = np.random.randint(1e4, size=(self.args.epochs, ))
    
    def test(self, epoch = 0):
        list_acc, list_loss = [], []
        self.global_model.eval()
        
        for c in range(self.args.num_users):
            acc, loss = self.train_agents[c].inference(model=self.global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy = sum([list_acc[key] * self.group_ws[key] for key in self.group_ws.keys()])
        train_loss = sum([list_loss[key] * self.group_ws[key] for key in self.group_ws.keys()])

        test_accuracy, test_loss = self.test_agent.inference(model=self.global_model)

        tqdm.write('At round {} accuracy: {}'.format(epoch, test_accuracy))
        tqdm.write('At round {} loss: {}'.format(epoch, test_loss))
        tqdm.write('At round {} training accuracy: {}'.format(epoch, train_accuracy))
        tqdm.write('At round {} training loss: {}'.format(epoch, train_loss))
        
    def train_one_round(self, model, epoch):
        local_weights = []
        model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        for idx in idxs_users:
            w, _  = self.train_agents[idx].update_weights(model=copy.deepcopy(model))
            local_weights.append((copy.deepcopy(w), self.group_ws[idx]))
        return local_weights
    
    def agg(self, updated_local_weights, prev_weights):
        agg_weight = average_weights(updated_local_weights)
        global_difference = copy.deepcopy(agg_weight)
        for key, value in agg_weight.items():
            global_difference[key] = value - prev_weights[key]
        return agg_weight, global_difference
    
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
                updated_local_weights = self.train_one_round(self.global_model, epoch)
                agg_weight, global_difference = self.agg(updated_local_weights, prev_weights)
            
            if self.args.algorithm == 'Upcycled' and (epoch+1) % 2 ==0 :
                for key, value in agg_weight.items():
                    if 'num_batches_tracked' in key:
                        continue
                    agg_weight[key] += self.args.upcycled_param * global_difference[key]
                    
            self.global_model.load_state_dict(agg_weight)
            self.test(epoch+1)