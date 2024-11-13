import torch
import numpy as np
import random
import os
import wandb

def set_seed(config):
    torch.manual_seed(config.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(config.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.TRAIN.RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.TRAIN.RANDOM_SEED)
    random.seed(config.TRAIN.RANDOM_SEED)

def save_model(config, model):
    output_path = os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    if not os.path.exists(config.MODEL.SAVED_DIR):
        os.makedirs(config.MODEL.SAVED_DIR)
    torch.save(model, output_path)

def wandb_model_log(config):
    model_path = os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    wandb.save(model_path)
    artifact = wandb.Artifact(name=f"{config.MODEL.MODEL_NAME}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)