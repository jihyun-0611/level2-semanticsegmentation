import torch
import numpy as np
import random
import os
import wandb
import pandas as pd

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
    torch.save(model.state_dict(), output_path)

def wandb_model_log(config):
    model_path = os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    wandb.save(model_path)
    artifact = wandb.Artifact(name=f"{config.MODEL.MODEL_NAME}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
def save_csv(config, rles, mode, epoch=None):
    mode_config = getattr(config, mode.upper())
    os.makedirs(mode_config.OUTPUT_DIR, exist_ok=True)
    csv_name = f'epoch{epoch}_' + mode_config.CSV_NAME if epoch is not None else mode_config.CSV_NAME
    output_path = os.path.join(mode_config.OUTPUT_DIR, csv_name)
    df = pd.DataFrame({
        "image_name": rles['image_names'],
        "class": rles['classes'],
        "rle": rles['rles'],
    })
    df.to_csv(output_path, index=False)
    print(f"{mode} results saved to {output_path}")