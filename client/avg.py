    

import torch
import copy
import time
from torch import nn
from torch.utils.data import DataLoader
from FedProxOptimizer import FedProxOptimizer
    
    

class LocalUpdate(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(dataset,
                                      batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def update_weights(self, model):
        model.train()
        epoch_loss = []
        
        #Non private LR and u: 1e-1-u-0.9 for iid; 5e-3-u-0.7 for 0.5_0.5; 1e-1-u-0.3 for 0_0; 5e-3-u-0.9 for 1_1
        #Output pert: 1e-1-u-0.3; 7e-2-u-0.3; 3e-2-u-0.5; 1e-2-u-0.1
        #obj pert: 1e-2-u-0.9; 1e-3-u-0.7; 1e-3-u-0.9; 1e-3-u-0.3
        optimizer = FedProxOptimizer(model.parameters(),
                                    lr=self.args.lr,
                                    mu=self.args.prox_param,
                                    momentum=self.args.momentum,
                                    nesterov=False,
                                    weight_decay=0)
        
            
        self.noise = None
        #Objective perturbation
        if self.args.alpha > 0:
            seed = time.time()
            torch.random.manual_seed(int((seed - int(seed)) * 1e3))
            grad_dire = []
            norm_sepe = 0.0
            for group in optimizer.param_groups:
                for param in group['params']:
                    grad_dire.append(torch.rand(param.shape, device=self.device)-0.5)
                    norm_sepe += torch.norm(grad_dire[-1])
            Gamma = torch.distributions.gamma.Gamma(self.args.feature_len, self.args.alpha)
            norm = Gamma.sample()
            norm.to(self.device)
            coef = norm / norm_sepe
#             print(norm, norm_sepe)
            for item in grad_dire:
                item *= coef
            self.noise = grad_dire

        early_stop = self.args.local_ep
        flag = True
        # sample = torch.rand(1, device=self.device)
        if torch.rand(1, device=self.device) < self.args.straggler:
            early_stop = int(torch.torch.rand(1) * self.args.local_ep)
            while early_stop == 0:
                early_stop = int(torch.torch.rand(1) * self.args.local_ep)
            flag = False

        for iter in range(self.args.local_ep):
            if iter >= early_stop:
                break
            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device, dtype=torch.float), labels.to(device=self.device, dtype=torch.long)
                
                inputs = inputs.flatten(start_dim=1)
                model.zero_grad()
                log_probs = model(inputs)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step(noise=self.noise)

        #Output perturbation
        w = copy.deepcopy(model.state_dict())
        if self.args.clip > 0 and self.args.alpha == 0:
            w = self.private(w)
        return w, flag

    def private(self, w):
        #clip
        def get_norm(w):
            norm = 0
            w_ls = []
            for key in w.keys():
                w_temp = w[key].flatten()
                w_ls.append(w_temp)
                norm += torch.norm(w_temp)
            w_total = torch.hstack(w_ls)
#             print(torch.norm(w_total))
#             return norm
            return torch.norm(w_total)
        norm = get_norm(w)
#         print(norm)
        if norm > self.args.clip:
            for key in w.keys():
                w[key] *= (self.args.clip / norm)
        #add noise
        seed = time.time()
        torch.random.manual_seed(int((seed - int(seed)) * 1e3))
        for key in w.keys():
            w[key] += torch.normal(0, self.args.sigma ** 2, size=tuple(w[key].shape), device=self.device)
        return w

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        cnt = 0
        for batch_idx, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device, dtype=torch.float), labels.to(device=self.device, dtype=torch.long)
            # Inference
            inputs = inputs.flatten(start_dim=1)
            outputs = model(inputs)
            # labels = labels.reshape(outputs.shape)

            batch_loss = self.criterion(outputs, labels)
            batch_loss *= labels.shape[0]
            cnt += labels.shape[0]
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.reshape(labels.shape)
            correct += torch.sum(pred_labels == labels).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss / cnt
#