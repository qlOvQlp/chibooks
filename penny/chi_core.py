# This is a simple framework for datasets fitting in DDP manner 
# author:   freya
# date:     oct.1.2023

import time
import torch
import logging
from typing import Optional
from chibooks.soul.utils import accuracy, accuracy_raw

logger = logging.getLogger("chibooks")
# should be overload:
class taskbook:
    def __init__(self) -> None:
        self.model = None
        self.loss_fn = None
        self.params = None
        self.log = logger.info
        # TODO delete __prepare

    def setup(self,model,loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        

    ## user overload setup interface
    ## set parameters to be optimised
    ## solve model.sp.xxx here
    def user_setup(self,cfg):
        if cfg.task == "fit":
            scaler = (cfg.dataset.train_batch_size * len(cfg.env.gpus)) / 256.
            self.params = [
                {
                    "params": self.model.module.parameters(),
                    "lr": cfg.fit.lr * scaler, 
                    "weight_decay": cfg.fit.wd 
                }
            ]
        elif cfg.task == "test":
            ## TODO user overload
            pass
        else:
            ## TODO user inference setup
            # prepare inference dataloader 
            pass 

    ## user overload train interface
    ## define loss calculation
    def train_step(self,batch, aux_info:Optional[dict]=None):
        x,y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y,y_hat)

        # return loss for update model
        return loss


    def validate_step(self,batch, aux_info:Optional[dict]=None):
        x,y = batch
        y_hat = self.model(x)
        top1,top5 = accuracy(y_hat,y,topk=(1,5))
        return {"top1":top1, "top5":top5}


    # optional
    def test_step(self,batch, aux_info:Optional[dict]=None):
        x,y = batch
        y_hat = self.model(x)
        top1,top5 = accuracy(y_hat,y,topk=(1,5))
        return {"top1":top1, "top5":top5}

    # optional
    def inference_step(self,batch,aux_info:Optional[dict]=None):
        # x,y = batch
        # y_hat = self.model(x)
        # return y_hat
        pass