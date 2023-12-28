import os
import json 
from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class ext_imagefolder(ImageFolder):
    def __init__(self, 
        root: str, 
        transform: Optional[Callable] = None, 

    ):
        super().__init__(root, transform)

        ## level 1:raw 2.simple 3:mid 4:detalied
        # self.cls_idx_fix, self.mega_label , self.cls_sample_amout= self.__aux_init(root,subset,threshold)


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        f_name = path.rsplit("/")[-1]

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, target, {"mega_fixed":mega_y, "f_name":f_name, "cls_amount":self.cls_sample_amount[target]}
        return sample, target, {"f_name":f_name}


class ext_imagefolder_dm():
    def __init__(self, 
        root: str,         
        img_size = [224,336]
    ):
        self.root = root
        self.img_size = img_size
        self.trans = trans.Compose([
            # trans.Resize(256,trans.InterpolationMode.BICUBIC),
            # trans.CenterCrop(self.img_size),
            trans.Resize(self.img_size,trans.InterpolationMode.BICUBIC),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])        

    def setup(self,stage:str)->None:
        if stage == "inference":
            self.inference_set = ext_imagefolder(
                root=self.root,
                transform=self.trans
            )

def get_ext_imagefolder_from_cfg(cfg):
    dm = ext_imagefolder_dm(
        root=cfg.dataset.root,
        img_size=cfg.dataset.img_size,
    )
    dm.setup(cfg.task)
    return dm