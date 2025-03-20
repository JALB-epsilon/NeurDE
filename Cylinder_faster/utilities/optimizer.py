import torch
from torch import optim


# Added optimizer_type
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
    

#add the scheduler_type
def get_scheduler(optimizer, scheduler_type, total_steps, config):
    if scheduler_type == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 1e-3),
            total_steps=total_steps,
            pct_start=config.get("pct_start", 0.3),
            div_factor=config.get("div_factor", 10),
            final_div_factor=config.get("final_div_factor", 100),
        )
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", total_steps // 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("eta_min", 0),
        )
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
        )
    elif scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1)
        )
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported.")

