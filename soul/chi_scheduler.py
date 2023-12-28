import math
import logging
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["warmup_cosine"]
logger = logging.getLogger("chibooks")

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    # get_lr : get lr for every group of parameterA
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return [base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            t = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [base_lr * (self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * t))) for base_lr in self.base_lrs]

def get_warmup_cosine_schd_from_cfg(cfg,optimizer):
    schd = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=cfg.fit.sp.warm_up_epochs,
        total_steps=cfg.fit.epochs,
        min_lr_ratio=cfg.fit.sp.min_lr_ratio,
    )
    return schd


def list_available_schedulers(return_dict=False):
    scheduler_dict = {
        "warmup_cosine": get_warmup_cosine_schd_from_cfg,
    }
    if return_dict:
        return scheduler_dict
    else:
        print(list(scheduler_dict.keys()))

## return scheduler class
def get_scheduler(cfg, optimizer):
    ## TODO scan and merge user lossfunc
    scheduler_dict = list_available_schedulers(True)
    scheduler = scheduler_dict[cfg.fit.lr_schd](cfg, optimizer)
    return scheduler