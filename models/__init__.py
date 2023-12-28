from .dinov2 import build_model_from_cfg as build_dinov2_model
from .tiny_vit import build_model_from_cfg as build_tinyvit_model
from .clip import build_clip_model_from_cfg as build_clip_model
from .resnet import build_model_from_cfg as build_resnet_model

def get_model_list():
    return {
        "tiny_vit":build_tinyvit_model,
        "dinov2":build_dinov2_model,
        "clip":build_clip_model,
        "resnet":build_resnet_model
    }



def build_model_from_cfg(cfgs):
    model_dict = get_model_list()
    if cfgs.model.name in list(model_dict.keys()):
        model = model_dict[cfgs.model.name](cfgs)
    else:
        print("model not defined ... ")
        ## TODO
    return model