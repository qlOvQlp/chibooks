from torch import optim as optim
# Modified for TinyViT

## dull
def build_optimizer(cfg, params):

    opt_lower = cfg.fit.optim.lower()
    optimizer = None
    
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(params=params)
    else:
        # TODO    
        pass
    return optimizer