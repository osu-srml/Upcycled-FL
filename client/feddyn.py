from client.avg import LocalUpdate as LocalBase
import torch
import copy
import time
from FedProxOptimizer import FedProxOptimizer
from utils import  trainable_params, vectorize

class LocalUpdate(LocalBase):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)
    
    
    def update_weights(self, model, nabla):
        # Set mode to train model
        model.train()
        #Non private LR and u: 1e-1-u-0.9 for iid; 5e-3-u-0.9 for 0.5_0.5; 5e-3-u-0.3 for 0_0; 1e-3-u-0.7 for 1_1
        #Output pert: 1e-1-u-0.1; 7e-2-u-0.5; 5e-3-u-0.5; 1e-2-u-0.3
        #obj pert: 5e-2-u-0.9; 5e-3-u-0.3; 5e-3-u-0.9; 1e-3-u-0.9
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
            
        vectorized_global_params = vectorize(trainable_params(model), detach=True)
        for iter in range(self.args.local_ep):
            if iter >= early_stop:
                break
            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device, dtype=torch.float), labels.to(device=self.device, dtype=torch.long)
                inputs = inputs.flatten(start_dim=1)
                model.zero_grad()
                log_probs = model(inputs)
                loss_ce = self.criterion(log_probs, labels)
                
                vectorized_curr_params = vectorize(trainable_params(model))
                loss_algo = self.args.dyn_alpha * torch.sum(
                    vectorized_curr_params
                    * (-vectorized_global_params + nabla)
                )
                loss = loss_ce + loss_algo
                
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    trainable_params(model), max_norm=self.args.max_grad_norm
                )
                
                optimizer.step(noise=self.noise)
                
        w = copy.deepcopy(model.state_dict())
        if self.args.clip > 0 and self.args.alpha == 0:
            w = self.private(w)
        
        model.load_state_dict(w)    
        return w, flag, trainable_params(model, detach=True)