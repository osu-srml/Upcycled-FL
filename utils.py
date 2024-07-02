

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os,sys
import json
import pickle
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from collections import  OrderedDict
from typing import List, Tuple, Union


def split_backup(dict_users, frac = 0.8):
    dict_back = {}
    dict_beta = {}
    for i in dict_users.keys():
        dict_users[i] = list(dict_users[i])
        end = max(int(frac * len(dict_users[i])), 1)
        dict_back[i] = dict_users[i][end:]
        dict_users[i] = dict_users[i][:end]
        dict_beta[i] = 0
    return dict_users, dict_back, dict_beta

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    print(args.dataset)

    print('load' + args.dataset)

    train_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/mytrain.pt")
    test_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/test/mytest.pt")
    user_group_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/user_groups.json")
    with open(user_group_path, 'rb') as inf:
        user_groups = json.load(inf)
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)

    train_datasets = {}
    for key, value in user_groups.items():
        train_datasets[int(key)] = TensorDataset(*train_dataset[np.array(value)])

    return train_datasets, test_dataset, user_groups, compute_group_weights(user_groups)

def get_dataset_clients(args):
    print('load' + args.dataset)
    train_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/mytrain.pt")
    test_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/test/mytest.pt")
    user_group_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/user_groups.json")
    with open(user_group_path, 'rb') as inf:
        user_groups = json.load(inf)
    train_dataset = torch.load(train_path)
    
    test_user_group_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/test/user_groups.json")
    with open(test_user_group_path, 'rb') as inf_test:
        test_user_groups = json.load(inf_test)
    
    test_dataset = torch.load(test_path)
    train_datasets = {}
    test_datasets = {}

    for key, value in user_groups.items():
        train_datasets[int(key)] = TensorDataset(*train_dataset[np.array(value)])
    for key, value in test_user_groups.items():
        test_datasets[int(key)] = TensorDataset(*test_dataset[np.array(value)])

    return train_datasets, test_datasets, user_groups, compute_group_weights(user_groups)


def compute_group_weights(user_groups):
    weights = {}
    sum = 0
    for key, value in user_groups.items():
        sum += len(value)
        weights[int(key)] = len(value)
    for key in weights.keys():
        weights[key] /= sum
    return weights

def average_weights(ls):
    """
    Returns the average of the weights.
    """
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


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Baseline     : {args.baseline}')
    print(f'    Algorithm     : {args.algorithm}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Upcycled     : {args.upcycled_param}')
    print(f'    Straggler     : {args.straggler}')
    print(f'    sigma     : {args.sigma}')
    print(f'    clip     : {args.clip}')
    print(f'    alpha     : {args.alpha}')

    return

def trainable_params(
    src,
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters

def vectorize(
    src, detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])