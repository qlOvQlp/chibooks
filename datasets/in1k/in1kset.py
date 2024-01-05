from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
import os
import torch
from torchvision.datasets.folder import default_loader


class IN1K(ImageFolder):
    def __init__(self, 
        root: str, 
        transform: Optional[Callable] = None, 
    ):
        super().__init__(root, transform)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        cls_name = self.classes[target]
        _,pos,seq,f_name = path.rsplit("/",3)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, {"f_name":f_name,"cls_name":cls_name}


class IN1K_DM:
    def __init__(self,cfg) -> None:
        self.root = cfg.dataset.root
        self.img_size = cfg.dataset.img_size
        self.train_trans = trans.Compose([
            trans.RandomResizedCrop(
                size = self.img_size,
                scale = (0.64,1)),
            trans.RandomHorizontalFlip(),
            trans.RandomGrayscale(),
            trans.ColorJitter(
                brightness=0.3,
                contrast=0.3),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])

        
        self.val_trans = trans.Compose([
            trans.Resize(self.img_size,trans.InterpolationMode.BICUBIC),
            trans.CenterCrop(self.img_size),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
        

    def prepare_data(self) -> None:
        ## download data
        ## unzip data
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_set = IN1K(os.path.join(self.root,"train"),self.train_trans)
            self.val_set = IN1K(os.path.join(self.root,"val"),self.val_trans)
        elif stage == 'test':
            self.test_set = IN1K(os.path.join(self.root,"val"),self.val_trans)
    
    
def get_in1k_from_cfg(cfg):
    in1k_set = IN1K_DM(cfg)
    in1k_set.setup(cfg.task)

    return in1k_set