import torch
from torch import optim

def dispatch_optimizer(model, lr=0.001, optimizer_type='AdamW'):  # Added optimizer_type

    if isinstance(model, torch.nn.Module):
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type == 'AdaBelief':
            from adabelief_pytorch import AdaBelief 
            optimizer = AdaBelief(model.parameters(), lr=lr, eps=1e-8, rectify=False)
        elif optimizer_type == 'Lion':
            from lion_pytorch import Lion 
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-5)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else: #default
            optimizer = optim.Adam(model.parameters(), lr=lr)

        return optimizer

    elif isinstance(model, list):
        optimizers = []
        if optimizer_type == 'AdamW':
            optimizers = [optim.AdamW(model[i].parameters(), lr=lr) for i in range(len(model))]
        elif optimizer_type == 'AdaBelief':
            from adabelief_pytorch import AdaBelief
            optimizers = [AdaBelief(model[i].parameters(), lr=lr, eps=1e-8, rectify=False) for i in range(len(model))]
        elif optimizer_type == 'Lion':
            from lion_pytorch import Lion
            optimizers = [Lion(model[i].parameters(), lr=lr, weight_decay=1e-2) for i in range(len(model))]
        elif optimizer_type == 'SGD':
            optimizers = [optim.SGD(model[i].parameters(), lr=lr, momentum=0.9) for i in range(len(model))]
        else: #default
            optimizers = [optim.Adam(model[i].parameters(), lr=lr) for i in range(len(model))]
        return optimizers

def calculate_relative_error(pred, target):
    return torch.norm(pred - target) / torch.norm(target)