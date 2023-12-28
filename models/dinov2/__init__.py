# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os 
import torch
from . import vision_transformer as vits
from .cls_head import LinearClassifier

import logging
logger = logging.getLogger("chibooks")

def build_head_from_cfg(embed_dim,cfg):
    in_dim = (cfg.model.head_use_n_blocks+(1 if cfg.model.head_use_avgpool else 0)) * embed_dim
    head = LinearClassifier(in_dim,
                            use_n_blocks=cfg.model.head_use_n_blocks,
                            use_avgpool=cfg.model.head_use_avgpool,
                            num_classes=cfg.model.head_cls_num)
    return head

def load_pretrained_weights(model, pretrained_weights):
    ## TODO only load "teacher" weight in dino/dinov2
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    if "head" in pretrained_weights:
        state_dict = {f"linear.{k}": v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def build_model_from_cfg(cfg):
    if cfg.model.arch == "vit_giant2":
        _ffn_layer = "swiglufused"
    else:
        _ffn_layer = "mlp"

    if "vit" in cfg.model.arch:
        vit_kwargs = dict(
            img_size=518,
            patch_size=cfg.model.patch_size,
            init_values=1.0e-05,
            ffn_layer=_ffn_layer,
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        backbone = vits.__dict__[cfg.model.arch](**vit_kwargs)
        if os.path.isfile(cfg.model.backbone_weights):
            load_pretrained_weights(backbone,cfg.model.backbone_weights)
        if cfg.model.head_cls_num > 0:
            head = build_head_from_cfg(backbone.embed_dim,cfg)
            if cfg.model.head_weights is not None and os.path.isfile(cfg.model.head_weights):  
                load_pretrained_weights(head, cfg.model.head_weights)  
            else:  
                logger.info("head_weights is not defined, use zero init ... ")
        else:
            head = torch.nn.Identity()
        backbone.mask_token = None
        
        class dino_model(torch.nn.Module):
            def __init__(self, backbone, head) -> None:
                super().__init__()
                self.backbone = backbone
                self.head =head
                self.head_use_avgpool = cfg.model.head_use_avgpool
                self.head_use_n_blocks = cfg.model.head_use_n_blocks

            def forward(self,x):
                fe = self.backbone.get_intermediate_layers(x,n=cfg.model.head_use_n_blocks,return_class_token=True)
                pred = self.head(fe)
                return pred
        
        model = dino_model(backbone,head)
    return model

