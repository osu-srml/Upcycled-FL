    
from client.avg import LocalUpdate as LocalBase
import torch
import copy
import time
from torch import nn
from torch.utils.data import DataLoader
from FedProxOptimizer import FedProxOptimizer
    
class LocalUpdate(LocalBase):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(dataset,
                                      batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []
        self.client_model_delta = copy.deepcopy(model)
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
            
        
        sd_model_delta = self.client_model_delta.state_dict()
        for name, param in model.named_parameters():
            sd_model_delta[name] = model.state_dict()[name] - sd_model_delta[name] 
        
        
        return w, flag, sd_model_delta


