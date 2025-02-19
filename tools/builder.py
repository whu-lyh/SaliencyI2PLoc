import os

import torch
import torch.optim as optim

from SaliencyI2PLoc.build import build_model_from_cfg
from datasets.BaseDataset import PairwiseDataset
from tools.schedulers import (CosineAnnealingWithWarmupLR, CosineLRScheduler,
                              GradualWarmupScheduler, build_lambda_sche,
                              build_warmup_const_sche, build_warmup_cos_sche,
                              build_warmup_linear_sche)
from utils.logger import print_log
from utils.misc import worker_init_fn


def dataloader_builder(args, dataset, mode:str="none", collate_fn=None):
    """
        Dataloader builder, modified to better adapt the retrieval dataset
    """
    # preset before update the dataset
    batch_size = dataset.batch_size
    if dataset.mode != 'train':
        if mode == "pairwise": # specifically designed for validation and test 
            dataset = PairwiseDataset(dataset.qImages, dataset.qPcs, dataset.dbImages, dataset.dbPcs, 
                                      points_num=dataset.points_num, point_cloud_proxy=dataset.point_cloud_proxy,
                                      transform_img=dataset.transform if dataset.point_cloud_proxy == "points" else dataset.transform_proxy)
        else:
            raise ValueError('mode choice is necessary!')
    # active shuffle when train model
    shuffle = False
    drop_last = False
    sampler = None
    if dataset.mode == 'train':
        shuffle = True
        drop_last = True

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle, 
                                            collate_fn=collate_fn,
                                            num_workers=int(args.num_workers),
                                            worker_init_fn=worker_init_fn, 
                                            pin_memory=True,
                                            persistent_workers=True,
                                            prefetch_factor=int(args.num_workers/2),
                                            drop_last=drop_last)
    return dataset, sampler, dataloader

def model_builder(config):
    """
        build dataset from yaml config file, share registry build function with dataset builder
    """
    model = build_model_from_cfg(config)
    return model

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
        Add weight decay to different layers
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}] 

def build_opti_sche(base_model, config, param_groups:list=None):
    """
        Build optimizer and scheduler together
    """
    if config.model.baseline and config.model.mode == "vanilla":
        # Assign different learning rate to different parts for baseline model
        optimizer = optim.SGD(param_groups[0], **config.optimizer.kwargs)
        scheduler = optim.lr_scheduler.StepLR(optimizer, **config.scheduler.kwargs)
        # here the scheduler will run same as the baseline custom scheduler
        optimizer3D = optim.Adam(param_groups[1], lr=param_groups[1][0]["lr"])
        scheduler3D = optim.lr_scheduler.StepLR(optimizer3D, step_size=2, gamma=0.8, last_epoch=-2) # step_size=2, gamma=0.8
        return optimizer, optimizer3D, scheduler, scheduler3D
    
    if param_groups is None:
        param_groups = filter(lambda p: p.requires_grad, base_model.parameters())
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        # param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(param_groups, nesterov=False, **opti_config.kwargs)
    else:
        raise NotImplementedError()
    # scheduler
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'CosineAnnealingLR': # also with warmup, same as CosLR
        scheduler = CosineAnnealingWithWarmupLR(optimizer, sche_config.kwargs)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'ReduceLROnPlateau': # reduce the lr when the loss comes to plateau status
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'WarmupConstantSchedule':
        scheduler = build_warmup_const_sche(optimizer, sche_config.kwargs)
    elif sche_config.type == 'WarmupLinearSchedule':
        scheduler = build_warmup_linear_sche(optimizer, sche_config.kwargs)
    elif sche_config.type == 'WarmupCosineSchedule':
        scheduler = build_warmup_cos_sche(optimizer, sche_config.kwargs)
    else:
        raise NotImplementedError()
    # add warmup trick when the specific optimizer and scheduler is applied
    if sche_config.type in ["LambdaLR", "StepLR"]:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, # scale to 2.0 times of the input lr
                                           total_epoch=config.max_epoch, 
                                           after_scheduler=scheduler)
    return optimizer, scheduler

def resume_model(base_model, args, logger=None):
    """
        Resume model weights
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path} ...', logger=logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path} ...', logger=logger)
    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)
    # parameter
    start_epoch = state_dict['epoch']
    best_metrics = state_dict['best_metrics']
    print_log(f'[RESUME INFO] resume ckpts at {start_epoch} epoch (best_score = {str(best_metrics):s})', logger=logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger=None):
    """
        Resume the optimizer
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path} ...', logger=logger)
        return
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path} ...', logger=logger)
    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    optimizer.load_state_dict(state_dict['optimizer'])

def resume_optimizer_refactor(optimizer, optimizer3D, args, logger=None):
    """
        Resume the optimizer, and optimizer3D
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path} ...', logger=logger)
        return
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path} ...', logger=logger)
    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    optimizer.load_state_dict(state_dict['optimizer'])
    optimizer3D.load_state_dict(state_dict['optimizer3D'])

def resume_scaler(scaler, args, logger=None):
    """
        Resume the scaler, mainly for mixup precision training
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path} ...', logger=logger)
        return
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path} ...', logger=logger)
    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    scaler.load_state_dict(state_dict["scaler"])

def unwrap_dist_model(args, model):
    if args.gpu is not None: # do not use all gpus, specific single GPU
        return model
    else:
        return model.module

def save_checkpoint(args, base_model, optimizer, scaler, epoch, metrics, best_metrics, not_improved, prefix, logger=None):
    """
        Save the checkpoint at the current epoch at master GPU.
    """
    if args.local_rank == 0:
        torch.save({'base_model' : unwrap_dist_model(args, base_model).state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    'epoch' : epoch,
                    'metrics': metrics if metrics is not None else dict(),
                    'best_metrics': best_metrics,
                    'not_improved': not_improved,
                }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)

def save_checkpoint_refactor(args, base_model, best_metrics, optimizer=None, optimizer3D=None, 
                             epoch=None, metrics=None, logger=None, prefix="ckpt-best"):
    """
        Save the checkpoint at the current epoch at master GPU. suited for seperate optimizer and scheduler
    """
    if args.local_rank == 0:
        torch.save({'base_model' : unwrap_dist_model(args, base_model).state_dict(),
                    'optimizer' : optimizer.state_dict() if optimizer is not None else None,
                    'optimizer3D' : optimizer3D.state_dict() if optimizer3D is not None else None,
                    'epoch' : epoch,
                    'metrics': metrics if metrics is not None else dict(),
                    'best_metrics': best_metrics,
                }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)

def load_model(base_model, ckpt_path, logger=None):
    """
        Resume from a checkpoint.
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s ...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path} ...', logger=logger)
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict=True)
    # update epoch number and metrics
    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'Ckpts at {epoch} epoch (performance = {str(metrics):s})', logger=logger)
    return epoch