import os
import random
import re
import socket
from typing import Union

import numpy as np
import random
import sys
import torch
import torch.distributed as dist
import logging 
import torch.backends.cudnn as cudnn

logger = logging.getLogger("chibooks")


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_print_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.gpus[args.rank])
    print('|| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    dist.barrier()
    setup_print_for_distributed(args.rank == 0)
    fix_random_seeds(args.seed)



def gather_tensor_to_main(inp:torch.Tensor,concat:bool=False):
    world_size = get_world_size()
    gather_list = [
        torch.zeros_like(inp) for _ in range(world_size)
    ]
    dist.gather(inp,
                gather_list if is_main_process else None,
                dst=0)
    if is_main_process:
        if concat:
            return torch.concat(gather_list,dim=-1)
        else:
            return gather_list
    else:
        return None

def gather_dict_to_main(obj:dict,merge:bool=False):
    ## TODO check 
    world_size = get_world_size()
    gather_obj_list = [None for _ in range(world_size)]

    dist.gather_object(obj,
                gather_obj_list if is_main_process() else None,
                dst=0)
    
    ## merge : dicts should be like
    # # rank 0:
    # {
    #     "a":[1,2,3],
    #     "b":[4,5,6]
    # }
    # # rank 1:
    # {
    #     "a":[11,22,33],
    #     "b":[44,55,66]
    # }
    ## merge res:
    # {
    #     "a":[1,2,3,11,22,33],
    #     "b":[4,5,6,44,55,66]
    # }

    if is_main_process():
        res_dict = {k:[] for k in obj.keys()}
        if merge:
            for _dict in gather_obj_list:
                for k,v in _dict.items():
                    res_dict[k].extend(v)
            return res_dict
        else:
            return gather_obj_list
    else:
        return None    