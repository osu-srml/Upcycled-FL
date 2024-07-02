from client.avg import LocalUpdate as LocalBase

import torch
import copy
import time
from FedProxOptimizer import FedProxOptimizer
from utils import  trainable_params

class LocalUpdate(LocalBase):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)
    
    
    def update_weights(self, model, c_global, c_local):
        model.train()
        prev_model = copy.deepcopy(model)
        
        optimizer = FedProxOptimizer(model.parameters(),
                                        lr=self.args.lr,
                                        mu=0,
                                        momentum=self.args.momentum,
                                        nesterov=False,
                                        weight_decay=0)
            
        self.noise = None
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
            for item in grad_dire:
                item *= coef
            self.noise = grad_dire

        early_stop = self.args.local_ep
        flag = True
        if torch.rand(1, device=self.device) < self.args.straggler:
            early_stop = int(torch.torch.rand(1) * self.args.local_ep)
            while early_stop == 0:
                early_stop = int(torch.torch.rand(1) * self.args.local_ep)
            flag = False
        count=0
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
                
                # scaffold
                for param, c, c_i in zip(
                    trainable_params(model),
                    c_global,
                    c_local,
                ):
                    param.grad.data += c - c_i
                
                optimizer.step(noise=self.noise)
                count+=1

        len_dataloader  = len(self.trainloader)
        w = copy.deepcopy(model.state_dict())
            
        
        # update local control variate
        with torch.no_grad():
            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for x, y_i in zip(trainable_params(prev_model), trainable_params(model)):
                y_delta.append(y_i - x)

            # compute c_plus
            coef = 1 / (count * self.args.lr)
            for c, c_i, x, y_i in zip(
                c_global,
                c_local,
                trainable_params(prev_model),
                trainable_params(model),
            ):
                c_plus.append(c_i - c + coef * (x - y_i))

            # compute c_delta
            for c_p, c_l in zip(c_plus, c_local):
                c_delta.append(c_p - c_l)

            c_local = c_plus 
        
        if self.args.clip > 0 and self.args.alpha == 0:
            w = self.private(w)
            
        return w, flag, c_plus, c_delta
