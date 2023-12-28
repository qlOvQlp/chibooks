from .cliploss import ClipLoss
from .soft_celoss import Soft_CrossEntropy

def init_ClipLoss(cfg):
    loss_fn = ClipLoss(
        rank=cfg.env.rank,
        world_size=cfg.env.world_size,
    )
    return loss_fn

def init_Soft_CrossEntropy(cfg):
    ## TODO
    return Soft_CrossEntropy()