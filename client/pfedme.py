from client.avg import LocalUpdate as LocalBase

import torch
import copy
import time
from typing import Dict, List
from FedProxOptimizer import FedProxOptimizer
from torch.optim import Optimizer
from utils import  trainable_params

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, local_parameters: List[torch.nn.Parameter], noise=None):
        group = None
        i = 0
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                x = group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data 
                )
                if noise is not None:
                    x.add_(1, noise[i])
                    i += 1
                param_p.data = param_p.data - x
                
class LocalUpdate(LocalBase):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)
    
    def update_weights(self, model):
        model.train()
        local_model = copy.deepcopy(model)
        local_parameters = trainable_params(local_model, detach=True)
        epoch_loss = []
        optimizer = FedProxOptimizer(model.parameters(),
                                        lr=self.args.per_lr,
                                        mu=0,
                                        momentum=self.args.momentum,
                                        nesterov=False,
                                        weight_decay=0)
        
        optimizer = pFedMeOptimizer(
            trainable_params(model),
            self.args.lr,
            self.args.pfedme_lambda,
            self.args.pfedme_mu,
        )

            
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

        for iter in range(self.args.local_ep):
            if iter >= early_stop:
                break
            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device, dtype=torch.float), labels.to(device=self.device, dtype=torch.long)
                
                inputs = inputs.flatten(start_dim=1)
                for _ in range(self.args.k): 
                    model.zero_grad()
                    log_probs = model(inputs)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step(local_parameters, noise=self.noise)
                    
                for param_p, param_l in zip(
                    trainable_params(model), local_parameters   
                ):
                    param_l.data = (
                        param_l.data
                        - self.args.pfedme_lambda
                        * self.args.lr
                        * (param_l.data - param_p.data)
                    )

        local_model_params = [param for param in local_model.parameters() if param.requires_grad]
        assert len(local_model_params) == len(local_parameters), "Mismatch in number of parameters"
        for param, local_param in zip(local_model_params, local_parameters):
            param.data.copy_(local_param)

        w = copy.deepcopy(local_model.state_dict())
        if self.args.clip > 0 and self.args.alpha == 0:
            w = self.private(w)
            
        w_p = copy.deepcopy(model.state_dict())
        if self.args.clip > 0 and self.args.alpha == 0:
            w_p = self.private(w_p)
            
            
        return w, flag, w_p