from .in1kset import get_in1k_from_cfg
from .terra_inc.terra_inc import get_terrainc_from_cfg
from .wilds_iwildcam.aux_wilds_iwildcam import get_wilds_iwildcam_from_cfg
from .common_inference.ext_imagefolder import get_ext_imagefolder_from_cfg

from torch.utils.data import DataLoader
from .sampler import MyDistributedSampler

import logging


__all__ = ["in1k","terra_inc","wilds_iwildcam","common_imagefolder"]
logger = logging.getLogger("chibooks")


def get_dataset_list():
    return {
        "in1k": get_in1k_from_cfg,
        "terra_inc": get_terrainc_from_cfg,
        "wilds_iwildcam": get_wilds_iwildcam_from_cfg,
        "common_imagefolder": get_ext_imagefolder_from_cfg
    }

def setup_dataloader_from_cfg(cfg):
    dataset_dict = get_dataset_list()
    if cfg.dataset.name in dataset_dict.keys():
        dataset = dataset_dict[cfg.dataset.name](cfg)
    else:
        logger.error("dataset not defined ... ")
        
        ## TODO
        # stop code
    if cfg.task == "fit":
        # generate dataloder from dataset:
        train_sampler = MyDistributedSampler(dataset.train_set,
             shuffle=True, drop_last= False, padding=True, seed=cfg.env.seed
        )
        train_dataloader = DataLoader(
            dataset=dataset.train_set,
            sampler=train_sampler,
            batch_size=cfg.dataset.train_batch_size,
            num_workers=cfg.env.num_workers,
            pin_memory=True,
        )
        if cfg.fit.val_freq > 0:
            val_sampler = MyDistributedSampler(dataset.val_set,
                shuffle=False, drop_last= False, padding=False,seed=cfg.env.seed
            )

            val_dataloader = DataLoader(
                dataset=dataset.val_set,
                sampler=val_sampler,
                batch_size=cfg.dataset.val_batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True
            )

            logger.info(f"setup train/val loaders with {len(dataset.train_set)}/{len(dataset.val_set)} samples")
            return train_dataloader, val_dataloader
        logger.info(f"setup train loaders with {len(dataset.train_set)} samples")
        return train_dataloader

    elif cfg.task == "test":
        test_sampler = MyDistributedSampler(dataset.test_set,
             shuffle=False, drop_last= False, padding=False,seed=cfg.env.seed
        )
        test_dataloader = DataLoader(
            dataset=dataset.test_set,
            sampler=test_sampler,
            batch_size=cfg.dataset.test_batch_size,
            num_workers=cfg.env.num_workers,
            pin_memory=True
        )
        logger.info(f"setup test loaders with {len(dataset.test_set)} samples")
        return test_dataloader
    else:
        inference_sampler = MyDistributedSampler(dataset.inference_set,
             shuffle=False, drop_last= False, padding=False,seed=cfg.env.seed
        )
        inference_dataloader = DataLoader(
            dataset=dataset.inference_set,
            sampler=inference_sampler,
            batch_size=cfg.dataset.inference_batch_size,
            num_workers=cfg.env.num_workers,
            pin_memory=True
        )
        logger.info(f"setup inference loaders with {len(dataset.test_set)} samples")
        return inference_dataloader