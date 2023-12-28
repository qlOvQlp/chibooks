from torch.nn.modules import loss
import torch.nn.functional as F


from .loss_from_freya import init_ClipLoss, init_Soft_CrossEntropy

    
def list_available_lossfn(return_dict=False):
    sp_loss_dict = {"Soft_CrossEntropy": init_Soft_CrossEntropy,
                            "Cliploss": init_ClipLoss}
    loss_dict = {k:loss.__dict__[k] for k in loss.__all__}
    # loss_dict.update(loss_fn_in_this_file)
    if return_dict:
        return loss_dict, sp_loss_dict
    else:
        print(list(loss_dict.keys()))

## return loss_fn class
def get_loss_fn(cfg):
    ## TODO scan user lossfunc
    loss_dict, sp_loss_dict = list_available_lossfn(True)
    if cfg.fit.loss_fn in sp_loss_dict.keys():
        loss_fn = sp_loss_dict[cfg.fit.loss_fn](cfg)
    else:
        # ce loss or regular loss fn
        loss_fn = loss_dict[cfg.fit.loss_fn]()
    ## TODO 

    return loss_fn