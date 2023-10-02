import os
import sys
import torch
import numpy as np
import logging
import argparse
import wandb
import random
import string

# seed setting
def set_seed(seed:int=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# logger setting
def get_logger(logger_name:str=None):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    
    return logger

def init_wandb(wandb_dir="./wandb", project_name="project", run_name:str=None, args:argparse.Namespace=None):
    if run_name is None:
        run_name = get_run_name() # random name generation

    wandb.init(dir=wandb_dir, project=project_name, name=run_name, config=args.__dict__)

def get_run_name(size=12):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(size))
