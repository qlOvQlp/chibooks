import os
import time
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf

import chibooks.distributed as chi_dist
import chibooks.logging as chi_log
import chibooks.datasets as chi_dat
import chibooks.models as chi_models
import chibooks.soul as chi_sol
import json

from chibooks.penny import taskbook
from sklearn.metrics import f1_score

def __main_worker(rank,cfg,task:taskbook):
    ## setup dist env and logger
    cfg.env.rank += rank
    chi_dist.init_distributed_mode(cfg.env)
    logger = chi_log.setup_logging(os.path.join(cfg.env.log_root,"logs"),name="chibooks")

    ## build model
    logger.info("set up model.")
    model = chi_models.build_model_from_cfg(cfg)
    model.cuda()

    logger.info("ddp OK.")
    if cfg.task == "fit":

        start_iter = 0
        logger.info("Starting training from iteration {}".format(start_iter))
        ## setup ddp model and dataloader
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[cfg.env.gpus[rank]])
        train_loader,val_loader = chi_dat.setup_dataloader_from_cfg(cfg)
        
        loss_fn = chi_sol.get_loss_fn(cfg)

        task.setup(model,loss_fn)
        # setup model parameter group
        task.user_setup(cfg)

        optimizer = chi_sol.build_optimizer(cfg, task.params)
        lr_scheduler= chi_sol.get_scheduler(cfg, optimizer)

        ## setup logger
        metric_log_file = os.path.join(cfg.env.log_root,"logs","training_metrics.log")
        metric_logger = chi_log.MetricLogger(delimiter="  ", output_file=metric_log_file)
        val_metric_logger = chi_log.MetricLogger(delimiter="  ")


        for epoch in range(cfg.fit.epochs):

            ## reset train_logger
            metric_logger.add_meter('lr', chi_log.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            train_loader.sampler.set_epoch(epoch)
            if cfg.fit.sp.freeze_backbone:
                model.module.head.train()
            else:
                model.train()

            for tmp_batch in metric_logger.log_every(train_loader, 20, header=f"Training [{epoch:3d}/{cfg.fit.epochs}] epoch"):
                
                inp = tmp_batch[0].cuda(non_blocking=True)
                target = tmp_batch[1].cuda(non_blocking=True)
                batch = (inp,target) if len(tmp_batch)<3 else (inp,target,tmp_batch[2:])
                loss,loss_aux_metric = task.train_step(batch,aux_info={"epoch":epoch})
                optimizer.zero_grad()
                loss.backward()
                # step
                optimizer.step()
                # make sure GPU task finished
                torch.cuda.synchronize()
                metric_logger.update(loss=loss.item()) ## for tensor stuff
                metric_logger.update(**loss_aux_metric) ## other value
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                

            # gather the stats from all processes after train an epoch
            metric_logger.synchronize_between_processes()

            # update lr by epoch, not by iter
            lr_scheduler.step()

            # check val peroid 
            if cfg.fit.val_freq > 0:
                if epoch % cfg.fit.val_freq == 0 or epoch == cfg.fit.epochs - 1:

                    _pred_ = []
                    _target_ = []

                    with torch.no_grad():
                        model.eval()
                        for tmp_batch in val_metric_logger.log_every(val_loader, 20, header="Validating"):
                            inp = tmp_batch[0].cuda(non_blocking=True)
                            target = tmp_batch[1].cuda(non_blocking=True)
                            batch = (inp,target) if len(tmp_batch)<3 else (inp,target,tmp_batch[2:])
                            val_res_dict = task.validate_step(batch,aux_info=None)
                            val_metric_logger.meters["top1"].update(val_res_dict["top1"].item(),num=inp.size()[0])
                            val_metric_logger.meters["top5"].update(val_res_dict["top5"].item(),num=inp.size()[0])
                            _pred_.extend(val_res_dict["pred"])
                            _target_.extend(val_res_dict["target"])
                        val_metric_logger.synchronize_between_processes()
                        res_all =  chi_dist.gather_dict_to_main({"pred":_pred_,"target":_target_},merge=True)
                        if chi_dist.is_main_process():
                            macro_f1 = f1_score(res_all["target"],res_all["pred"],labels=torch.tensor(res_all["target"]).unique(),average="macro")
                            logger.info(f"macro_f1: {macro_f1}")
                        logger.info(f"top1: {val_metric_logger.meters['top1'].global_avg:.2f}%, top5: {val_metric_logger.meters['top5'].global_avg:.2f}%")
            # check ckpt peroid 
            if cfg.fit.ckpt_freq > 0:
                if epoch % cfg.fit.ckpt_freq == 0 or epoch == cfg.fit.epochs - 1:
                    # utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
                    if chi_dist.is_main_process():
                        torch.save(model.state_dict(),os.path.join(cfg.env.log_root,"ckpts",f"ep_{epoch}.pth"))

    elif cfg.task == "test":

        logger.info("Starting testing ... ")
        metric_logger = chi_log.MetricLogger(delimiter="  ",)

        ## setup ddp model and dataloader
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[cfg.env.gpus[rank]])
        test_loader = chi_dat.setup_dataloader_from_cfg(cfg)
        task.setup(model,None)
        task.user_setup(cfg)
        
        _pred_ = []
        _target_ = []
        with torch.no_grad():
            model.eval()
            for tmp_batch in metric_logger.log_every(test_loader, 20, header="testing"):

                inp = tmp_batch[0].cuda(non_blocking=True)
                target = tmp_batch[1].cuda(non_blocking=True)
                batch = (inp,target) if len(tmp_batch)<3 else (inp,target,tmp_batch[2:])
                test_res_dict = task.test_step(batch,aux_info=None)
                metric_logger.meters["top1"].update(test_res_dict["top1"].item(),num=inp.size()[0])
                metric_logger.meters["top5"].update(test_res_dict["top5"].item(),num=inp.size()[0])

                _pred_.extend(test_res_dict["pred"])
                _target_.extend(test_res_dict["target"])

            metric_logger.synchronize_between_processes()
            res_all =  chi_dist.gather_dict_to_main({"pred":_pred_,"target":_target_},merge=True)
            if chi_dist.is_main_process():
                macro_f1 = f1_score(res_all["target"],res_all["pred"],labels=torch.tensor(res_all["target"]).unique(),average="macro")
                logger.info(f"macro_f1: {macro_f1}")

            logger.info(f"total: {metric_logger.meters['top1'].total}, count:{metric_logger.meters['top1'].count}")
            logger.info(f"top1: {metric_logger.meters['top1'].global_avg:.2f}%, top5: {metric_logger.meters['top5'].global_avg:.2f}%")

    # inference on single GPU for test
    # inference is designed for small sizedata evaluate and get res not for acc or F1-score
    else:
        logger.info("Starting inferencing ... ")
        metric_logger = chi_log.MetricLogger(delimiter="  ",)

        ## setup ddp model and dataloader
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[cfg.env.gpus[rank]])
        inferencing_loader = chi_dat.setup_dataloader_from_cfg(cfg)
        task.setup(model,None)
        task.user_setup(cfg)
        
        _pred_ = []
        with torch.no_grad():
            model.eval()
            for tmp_batch in metric_logger.log_every(inferencing_loader, 20, header="inferencing"):

                inp = tmp_batch[0].cuda(non_blocking=True)
                target = tmp_batch[1].cuda(non_blocking=True)
                batch = (inp,target) if len(tmp_batch)<3 else (inp,target,tmp_batch[2:])
                
                test_res_dict = task.inference_step(batch,aux_info=None)

                _pred_.extend(test_res_dict["pred"])

            metric_logger.synchronize_between_processes()
            res_all =  chi_dist.gather_dict_to_main({"pred":_pred_,"target":_target_},merge=True)

            if chi_dist.is_main_process():
                macro_f1 = f1_score(res_all["target"],res_all["pred"],labels=torch.tensor(res_all["target"]).unique(),average="macro")
                logger.info(f"macro_f1: {macro_f1}")



def print_params(merged_config, default_config, prefix=''):
    for key, value in OmegaConf.to_container(merged_config, resolve=True).items():
        if isinstance(value, dict):
            print_params(OmegaConf.create(value), OmegaConf.create(default_config.get(key, {})), prefix=f'{prefix}{key}.')
        elif key in default_config and value != default_config[key]:
            print(f"{prefix}{key}: {default_config[key]} --> {value}")
        else:
            print(f"{prefix}{key}: {value}")

# chibooks entry function
# cfg: path to cfg file
# task: user overload to control fit_flow
def on_my_mark(*,cfg_path,task:taskbook):
    # Load the user config
    if os.path.isfile(cfg_path):
        user_config = OmegaConf.load(cfg_path)
    else:
        user_config = {}

    # Load the default config
    default_cfg_path = Path(__file__).parent.joinpath("configs/default.yaml")
    default_config = OmegaConf.load(default_cfg_path)
    

    # Merge the configs
    cfg = OmegaConf.merge(default_config, user_config)

    # Print all parameters and highlight updated ones
    print("\nParameters:")
    # print_params(cfg, default_config)
    print(cfg)

    # prepare env
    log_folder_suffix = f"[{time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())}]"
    log_root = os.path.join(cfg.env.log_root, log_folder_suffix)

    os.makedirs(f"{log_root}/ckpts",exist_ok=True)
    os.makedirs(f"{log_root}/logs",exist_ok=True)
    OmegaConf.save(cfg,f"{log_root}/logs/cfg.yaml")

    if isinstance(cfg.env.gpus, int):
        cfg.env.gpus = [i for i in range(cfg.env.gpus)]
    if cfg.task == "inference":
        cfg.env.gpus = [cfg.env.gpus[0]]


    cfg.env.rank = 0
    cfg.env.dist_url = 'tcp://127.0.0.1:14275'
    cfg.env.world_size = len(cfg.env.gpus)
    cfg.env.log_root = log_root


    # check whether task is overload
    # TODO 
    print(f"mutiprocess launching ... ")
    torch.multiprocessing.set_start_method("forkserver")
    torch.multiprocessing.spawn(__main_worker,
                            args = ((cfg),task),
                            nprocs= cfg.env.world_size)



if __name__ == "__main__":
    on_my_mark(cfg_path="",task=taskbook())