from collections import OrderedDict
from server.avg import Server as ServerBase
from tqdm import tqdm
import copy
import torch
from client.pfedme import LocalUpdate
from utils import get_dataset_clients
import numpy as np

class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.ClientUpdate = LocalUpdate
        
        self.train_datasets, self.test_dataset, self.user_groups, self.group_ws = get_dataset_clients(args)
        self.model_personalized = {}
        for c in range(args.num_users):
            self.model_personalized[c] = copy.deepcopy(self.global_model)
        
    def test(self, epoch = 0):
        list_acc, list_loss = [], []
        self.global_model.eval()
        
        # evaluate on the training set
        for c in range(self.args.num_users):
            self.model_personalized[c].eval()
            local_model = self.ClientUpdate(args=self.args, dataset=self.train_datasets[c], logger=self.logger)
            acc, loss = local_model.inference(model=self.model_personalized[c])
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy = sum([list_acc[key] * self.group_ws[key] for key in self.group_ws.keys()])
        train_loss = sum([list_loss[key] * self.group_ws[key] for key in self.group_ws.keys()])
        
        #evaluate on client test sets
        test_list_acc, test_list_loss = [], []
        for c in range(self.args.num_users):
            test_model = self.ClientUpdate(args=self.args, dataset=self.test_dataset[c], logger=self.logger)
            test_accuracy_item, test_loss_item = test_model.inference(model=self.model_personalized[c])
            test_list_acc.append(test_accuracy_item)
            test_list_loss.append(test_loss_item)
        test_accuracy = sum([test_list_acc[key]*self.group_ws[key] for key in self.group_ws.keys()])
        test_loss = sum([test_list_loss[key]*self.group_ws[key] for key in self.group_ws.keys()])
        
        #evaluate global model 
        global_test_list_acc, global_test_list_loss = [], []
        for c in range(self.args.num_users):
            global_test_model = self.ClientUpdate(args=self.args, dataset=self.test_dataset[c], logger=self.logger)
            global_test_accuracy_item, global_test_loss_item = global_test_model.inference(model=self.global_model)
            global_test_list_acc.append(global_test_accuracy_item)
            global_test_list_loss.append(global_test_loss_item)
        global_test_accuracy = sum([global_test_list_acc[key]*self.group_ws[key] for key in self.group_ws.keys()])
        global_test_loss = sum([global_test_list_loss[key]*self.group_ws[key] for key in self.group_ws.keys()])

        tqdm.write('At round {} accuracy: {}'.format(epoch, test_accuracy))
        tqdm.write('At round {} loss: {}'.format(epoch, test_loss))
        tqdm.write('At round {} global accuracy: {}'.format(epoch, global_test_accuracy))
        tqdm.write('At round {} global loss: {}'.format(epoch, global_test_loss))
        tqdm.write('At round {} training accuracy: {}'.format(epoch, train_accuracy))
        tqdm.write('At round {} training loss: {}'.format(epoch, train_loss))
        
    
    def train_one_round(self, model, epoch):
        local_weights = []
        model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = self.ClientUpdate(args=self.args, dataset=self.train_datasets[idx], logger=self.logger)
            w, flag, w_p = local_model.update_weights(model=copy.deepcopy(self.global_model))
            local_weights.append((copy.deepcopy(w), self.group_ws[idx]))
            self.model_personalized[idx].load_state_dict(w_p)
            
            if flag is False:
                continue
        return local_weights
    
        
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
            self.test(epoch+1) # in even step, personalized models wouldn't be updated if Upcycled