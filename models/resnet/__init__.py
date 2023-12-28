import logging
import torch
from .cls_head import LinearClassifier
from torchvision.models import resnet
logger = logging.getLogger("chibooks")

def build_model_from_cfg(cfg):
    if cfg.model.name == "resnet":
        if cfg.model.arch == "resnet18":
            model = resnet.resnet18(resnet.ResNet18_Weights)
        elif cfg.model.arch == "resnet50":
            model = resnet.resnet50(resnet.ResNet50_Weights.IMAGENET1K_V2)
    else:
        ## todo
        pass
    in_dim = model.fc.in_features
    model.fc = torch.nn.Identity()
    head = LinearClassifier(in_dim,cfg.model.sp.new_head)

    class resnet_model(torch.nn.Module):
        def __init__(self, backbone, head) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self,x):
            fe = self.backbone(x)
            pred = self.head(fe)
            return pred
        
    model = resnet_model(model,head)

    return model

def load_pretrained_weights(model, pretrained_weights=""):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = state_dict["model"]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))