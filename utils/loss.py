import torch.nn as nn

def get_loss(config):
    loss_name = config.TRAIN.LOSS.NAME
    loss_params = config.TRAIN.LOSS.get("PARAMS", {})  # 추가 파라미터
    return getattr(nn, loss_name)(**loss_params)