from .clip import *
from .model import CLIP
from .loss import ClipLoss

from typing import Any
import sys
import os

import logging 
logger = logging.getLogger("chibooks")

# default model loader

# vit-b/16
# vit-s/16
# vit-l/16
def build_model(*,arch_name="vit_base",patch_size=16,pretrained_weight=None):
    # def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    '''
        build dino vit model with default paras and web pretrained weight or local weights
    '''

    if arch_name in ["vit_base","vit_large"]:
        name = f"ViT-B/{patch_size}" if arch_name == "vit_base" else f"ViT-L/{patch_size}"
    elif arch_name == "vit_large_336":
        name = "ViT-L/14@336px"
    else:
        name = None

    if name in available_models():
        if os.path.isdir(pretrained_weight):
            model, transform = load(name,device="cpu",download_root=pretrained_weight)
    else:
        logger.info(f"Model {name} not found; available models = vit_base 32/16 vit_large 14")
        sys.exit(1)

    return model, transform

def build_clip_model_from_cfg(args):
    arch_name = args.model.arch
    patch_size = args.model.sp.patch_size
    pretrained_weight = args.model.sp.clip_weight_root
    
    model,trans = build_model(arch_name=arch_name,
                                    patch_size=patch_size,
                                    pretrained_weight=pretrained_weight)
    return model