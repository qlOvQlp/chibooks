from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
import os
import torch

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
            trans.Resize(256,trans.InterpolationMode.BICUBIC),
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
            self.train_set = ImageFolder(os.path.join(self.root,"train"),self.train_trans)
            self.val_set = ImageFolder(os.path.join(self.root,"val"),self.val_trans)
        elif stage == 'test':
            self.test_set = ImageFolder(os.path.join(self.root,"val"),self.val_trans)
    
    
def get_in1k_from_cfg(cfg):
    in1k_set = IN1K_DM(cfg)
    in1k_set.setup(cfg.task)

    return in1k_set