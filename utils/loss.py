import torch.nn as nn
import monai.losses

def get_loss(config):
    loss_name = config.TRAIN.LOSS.NAME
    loss_params = config.TRAIN.LOSS.get("PARAMS", {})  # 추가 파라미터

    if hasattr(monai.losses, loss_name):
        loss_class = getattr(monai.losses, loss_name)
    elif hasattr(nn, loss_name):
        loss_class = getattr(nn, loss_name)
    else:
        raise ValueError(f"monai 혹은 torch.nn 라이브러리에 {loss_class} 가 없습니다. loss명이 유효한 지 확인해주세요")

    return loss_class(**loss_params)
