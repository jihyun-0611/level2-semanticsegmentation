import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(config, optimizer):
    scheduler_name = config.TRAIN.SCHEDULER.NAME
    scheduler_params = config.TRAIN.SCHEDULER.get("PARAMS", {})
    return getattr(lr_scheduler, scheduler_name)(optimizer, **scheduler_params)