import math

from torch.optim.lr_scheduler import (CosineAnnealingLR, LambdaLR,
                                      ReduceLROnPlateau, SequentialLR,
                                      _LRScheduler)


def CosineAnnealingWithWarmupLR(optimizer, config):
    """
        optimizer (Optimizer): Wrapped optimizer.
        num_epochs (int): Total number of epochs.
        len_traindl (int): Total number of batches in the training Dataloader.
        warmup_epochs (int): Number of epochs for linear warmup. Default: 10.
        min_lr (float): Min learning rate. Default: 0.001.
        iter_scheduler (bool): If scheduler is used every iteration (True) or
                               every epoch (False). Default: True.
    """
    num_epochs = config.num_epochs
    len_traindl = config.len_traindl
    warmup_epochs = config.warmup_epochs
    min_lr = config.min_lr
    iter_scheduler = config.iter_scheduler
    # Declare iteration or epoch-wise scheduler
    if iter_scheduler:
        train_len = len_traindl
    else:
        train_len = 1 
    # Define Warmup scheduler
    scheduler1 = LambdaLR(optimizer, 
                          lambda it: (it+1)/(warmup_epochs*train_len))
    # Define Cosine Annealing Scheduler
    scheduler2 = CosineAnnealingLR(optimizer, 
                                   T_max=(num_epochs-warmup_epochs)*train_len, 
                                   eta_min=min_lr)
    # Combine both
    scheduler = SequentialLR(optimizer, 
                             schedulers=[scheduler1, scheduler2], 
                             milestones=[warmup_epochs*train_len])
    return scheduler


def build_lambda_sche(optimizer, config):
    """
        Custom scheduler where the learning decays at e/ decay step, the lowest decay rate is set
    """
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = LambdaLR(optimizer, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler


class ConstantLRSchedule(LambdaLR):
    """ 
        Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

def build_const_lambda_Sche(opti, config):
    return ConstantLRSchedule(optimizer=opti)


class WarmupConstantSchedule(LambdaLR):
    """ 
        Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
        # TODO USAGE?
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

def build_warmup_const_sche(opti, config):
    return WarmupConstantSchedule(optimizer=opti, warmup_steps=config.warmup_steps)


class WarmupLinearSchedule(LambdaLR):
    """ 
        Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def build_warmup_linear_sche(opti, config):
    return WarmupLinearSchedule(optimizer=opti, warmup_steps=config.warmup_steps, t_total=config.t_total)


class WarmupCosineSchedule(LambdaLR):
    """ 
        Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(self.min_lr, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def build_warmup_cos_sche(opti, config):
    return WarmupCosineSchedule(optimizer=opti, warmup_steps=config.warmup_steps, t_total=config.t_total)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                if type(self.after_scheduler) == ReduceLROnPlateau:
                    return self.after_scheduler._last_lr
                else:
                    return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            # deprecated in the future
            # if epoch is None:
            #     self.after_scheduler.step(metrics, None)
            # else:
            #     self.after_scheduler.step(metrics, epoch - self.total_epoch)
            self.after_scheduler.step(metrics, None)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == "__main__":
    import torch
    from easydict import EasyDict
    from timm.scheduler import CosineLRScheduler

    # custom learning rate decay scheduler
    sche_config = {
        "kwargs": {
            "lr_decay": 0.1,
            "decay_step": 2,
            "lowest_decay": 0.001,
        },
    }
    sche_config = EasyDict(sche_config)
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    scheduler = build_lambda_sche(optimizer, sche_config.kwargs)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("build_lambda_sche")
    print(lr)

    # decay the lr based on the initial_lr*80% every 2 epochs
    # this will keep the intial_lr smaller than 1 if the initial_lr is larger than 1
    # the lr will begin from initial_lr/100
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    optimizer.param_groups[0]['initial_lr'] = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-2)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("StepLR")
    print(lr)

    # add warmup trick(GradualWarmupSchedule) when the specific optimizer and scheduler is applied
    # unsupported to the sam optimizer
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, total_epoch=5, 
                                        after_scheduler=scheduler)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_lr()[0])
    print("GradualWarmupScheduler_StepLR")
    print(lr)

    # Decays the learning rate of each parameter group by linearly changing small
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.00006, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                  end_factor=0.2, total_iters=4000)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("LinearLR")
    print(lr)

    # add warmup trick(GradualWarmupSchedule) when the specific optimizer and scheduler is applied
    # unsupported to the sam optimizer
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, total_epoch=5, 
                                        after_scheduler=scheduler)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_lr()[0])
    print("GradualWarmupScheduler_LinearLR")
    print(lr)

    # Reduce learning rate when a metric has stopped improving
    # especilly suit for the loss as metric
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.1)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step(metrics=e)
        lr.append((scheduler._last_lr)[0])
    print("ReduceLROnPlateau")
    print(lr)

    # add warmup trick(GradualWarmupSchedule) when the specific optimizer and scheduler is applied
    # unsupported to the sam optimizer
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, total_epoch=5, 
                                        after_scheduler=scheduler)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step(epoch=e, metrics=e) # especially for ReduceLROnPlateau
        lr.append(scheduler.get_lr()[0])
    print("GradualWarmupScheduler_ReduceLROnPlateau")
    print(lr)

    # Cosine decay with restarts, this should be binded with SGD optimizer
    # more details at https://minibatchai.com/2021/07/09/Cosine-LR-Decay.html
    # warmup option is already coded inside
    sche_config = {
        "kwargs": {
            "t_initial": 10,
            "lr_min": 5e-6, # the lr will decay until this value
            "warmup_lr_init": 1e-5, # will warmup into the lr set at optimizer
            "warmup_t": 5,
            "warmup_prefix": True, # every new epoch number equals epoch = epoch - warmup_t
            "cycle_limit": 1,
            "t_in_epochs": True,
        },
    }
    sche_config = EasyDict(sche_config)
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=1e-3, momentum=0.9)
    scheduler = CosineLRScheduler(optimizer, **sche_config.kwargs)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step(e)
        lr.append((scheduler._get_lr(e))[0])
    print("CosineLRScheduler")
    print(lr)

    # same as the upper scheduler but implement by pytorch officically
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    optimizer.param_groups[0]['initial_lr'] = 0.1
    # clamp lr to eta_min at T_max, eta_min is the minimum lr when the lr clamps at T_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.01, last_epoch=-1)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(20):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("CosineAnnealingLR")
    print(lr)

    # add warmup trick(GradualWarmupSchedule) when the specific optimizer and scheduler is applied
    # unsupported to the sam optimizer
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, total_epoch=5, 
                                        after_scheduler=scheduler)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(20):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_lr()[0])
    print("GradualWarmupScheduler_CosineAnnealingLR")
    print(lr)

    # Linear warmup and then constant.
    # Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
    sche_config = {
        "kwargs": {
            "warmup_steps": 5,
            "last_epoch": -1,
        },
    }
    sche_config = EasyDict(sche_config)
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    scheduler = build_warmup_const_sche(optimizer, sche_config.kwargs)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(10):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("build_warmup_const_sche")
    print(lr)

    # Linear warmup and then linear decay.
    # Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    # Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    sche_config = {
        "kwargs": {
            "warmup_steps": 5,
            "t_total": 100,
            "last_epoch": -1,
        },
    }
    sche_config = EasyDict(sche_config)
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    scheduler = build_warmup_linear_sche(optimizer, sche_config.kwargs)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(sche_config.kwargs.t_total):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("build_warmup_linear_sche")
    print(lr)

    # Linear warmup and then cosine decay.
    # Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    # Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    sche_config = {
        "kwargs": {
            "warmup_steps": 5,
            "t_total": 100,
            "cycles": 0.5,
            "last_epoch": -1,
            "min_lr": 1e-6,
        },
    }
    sche_config = EasyDict(sche_config)
    optimizer = torch.optim.SGD(params=[torch.Tensor()], lr=0.1, momentum=0.9)
    scheduler = build_warmup_cos_sche(optimizer, sche_config.kwargs)
    lr = [optimizer.param_groups[0]['lr']]
    for e in range(sche_config.kwargs.t_total):
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    print("build_warmup_cos_sche")
    print(lr)