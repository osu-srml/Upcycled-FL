import argparse
import os
import time
from tqdm import tqdm
import importlib

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='avg')
    parser.add_argument('--algorithm', type=str, default="Upcycled",
                        help="if use Upcycled; otherwise use standard update ", choices=["Upcycled", "Vanilla"])
    parser.add_argument('--upcycled_param', type=float, default=0.3,
                        help="upcycled parameter, need to be tuned")
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients')
    parser.add_argument('--straggler', type=float, default=0,
                        help='straggler ratio')
    parser.add_argument('--dataset', type=str, default='synthetic_0_0', 
                        help="name of dataset, see the data folder")
    parser.add_argument('--gpu', type=int, default=0, 
                        help="To use cuda, set to a specific GPU ID.")
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed')
    
    #Clients parameters
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size")
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    
    # privacy parameters
    parser.add_argument('--sigma', type=float, default=1,
                        help="noise level in output perturbation; For baselines, Synthetic dataset: 1.0, Real: 0.27; For Upcycled, Synthetic dataset: 0.8, Real: 0.2; As Upcycled-FL doesn't use clients data in even epochs, it can achieve stricter privacy guarentees with less sigma.")
    parser.add_argument('--clip', type=float, default=0,
                        help='clip in output perturbation, default: 10')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha in objective perturbation; For baselines, Synthetic dataset: 10, FEMNIST: 100; sent140: 15; For Upcycled, Synthetic dataset: 20, FEMNIST: 200; sent140: 300')
    
    # baseline parameters
    parser.add_argument('--server_momentum', type=float, default=0.3,
                        help='server_momentum for fedavgm')
    parser.add_argument('--dyn_alpha', type=float, default=0.01,
                        help='alpha for feddyn')
    parser.add_argument('--max_grad_norm', type=float, default=10,
                        help='max_grad_norm for feddyn')
    parser.add_argument('--pfedme_lambda', type=float, default=15,
                        help='pfedme_lambda for pfedme')
    parser.add_argument('--pfedme_beta', type=float, default=1,
                        help='pfedme_beta for pfedme')
    parser.add_argument('--pfedme_mu', type=float, default=1e-3,
                        help='pfedme_mu for pfedme')
    parser.add_argument('--k', type=int, default=5,
                        help='personlization epoch')
    parser.add_argument('--per_lr', type=float, default=0.1,
                        help='personlization lr, same as local lr used in experiments')
    parser.add_argument('--yogi_lr', type=float, default=0.1,
                        help='fedyogi learning rate')
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='fedyogi parameter')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='fedyogi parameter')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='fedyogi parameter')
    parser.add_argument('--prox_param', type=float, default=0,
                        help="FedProx mu, need to be tuned for different datasets")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = args_parser()
    print(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    if(args.baseline != 'FedProx' and args.baseline != 'fedprox'): # FedProx optimizer is used for all baselines, it is necessary to keep 0 for other baselines to make it as the SGD optimizer.
        assert args.prox_param == 0
    else:
        assert args.prox_param > 0  # FedProx needs prox_param > 0
        
    Method = importlib.import_module('server.' + args.baseline)
    server = Method.Server(args)
    
    s1 = time.time()
    server.run()
    s2 = time.time()
    tqdm.write('Time: {}'.format(s2 - s1))
