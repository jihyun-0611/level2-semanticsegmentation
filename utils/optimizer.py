import torch.optim as optim

def get_optimizer(config, model):
    optimizer_name = config.TRAIN.OPTIMIZER.NAME
    optimizer_params = config.TRAIN.OPTIMIZER.get("PARAMS", {})
    return getattr(optim, optimizer_name)(
        model.parameters(),
        lr=config.TRAIN.LR,
        **optimizer_params
    )