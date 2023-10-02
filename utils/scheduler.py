import argparse
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.optim.lr_scheduer as lr_scheduler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_scheduler(optimizer:optim.Optimizer, scheduler_type:str=None, scheduler_params:dict=None):
    # transformers
    if scheduler_type == "get_cosine_schedule_with_warmup":
        # scheduler_params = {"num_warmup_steps":int, "num_training_steps":int}
        return get_cosine_schedule_with_warmup(optimizer, **scheduler_params)
    elif scheduler_type == "get_linear_schedule_with_warmup":
        # scheduler_params = {"num_warmup_steps":int, "num_training_steps":int}
        return get_linear_schedule_with_warmup(optimizer, **scheduler_params)
    else:
        raise ValueError("Unknown scheduler type: {}".format(scheduler_type))
    