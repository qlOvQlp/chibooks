from . import tiny_vit_base
from .cls_head import LinearClassifier
import os
import torch

import logging
logger = logging.getLogger("chibooks")

def build_model_from_cfg(cfg):

    if cfg.model.arch in tiny_vit_base.__dict__:
        model = tiny_vit_base.__dict__[cfg.model.arch]()

        if os.path.isfile(cfg.model.backbone_weight):
            load_pretrained_weights(model, cfg.model.backbone_weight)

    if cfg.model.sp.new_head is not None:
        ## replace origin imagenet head
        # in_dim = model.head.
        in_dim = model.head.in_features
        model.head = torch.nn.Identity()
        head = LinearClassifier(in_dim,cfg.model.sp.new_head)

        class tinyvit_model(torch.nn.Module):
            def __init__(self, backbone, head) -> None:
                super().__init__()
                self.backbone = backbone
                self.head =head

            def forward(self,x):
                fe = self.backbone(x)
                pred = self.head(fe)
                return pred
        
        model = tinyvit_model(model,head)

    return model


def load_pretrained_weights(model, pretrained_weights="/data/sls_dump/weights/tinyvit/tiny_vit_5m_22kto1k_distill.pth"):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = state_dict["model"]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
