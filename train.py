import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from utils.utils import set_seed, save_model, wandb_model_log
from utils.metrics import dice_coef
from data.dataset import XRayDataset
from data.augmentation import DataTransforms
from models import *

from config.config import Config

from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

import wandb

config = Config('config.yaml')

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)

            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    dice_dict = {f"val/{c}": d.item() for c, d in zip(config.DATA.CLASSES, dices_per_class)}
    
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(config.DATA.CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    val_loss = total_loss / cnt
    
    return avg_dice, dice_dict, val_loss

def train(model, data_loader, val_loader, criterion, optimizer, scheduler):
    run = wandb.init(
        project=config.WANDB.PROJECT_NAME,
        entity=config.WANDB.ENTITY, 
        name=config.WANDB.RUN_NAME, 
        notes=config.WANDB.NOTES, 
        tags=config.WANDB.TAGS, 
        config=config.WANDB.CONFIGS
    )
    wandb.watch(model, criterion, log="all", log_freq=config.WANDB.WATCH_STEP*len(data_loader))
    
    print(f'Start training..')
    
    best_dice = 0.
    scaler = GradScaler() if config.TRAIN.FP16 else None
    
    for epoch in range(config.TRAIN.EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            optimizer.zero_grad()

            if config.TRAIN.FP16:
                # FP16 사용 시
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 사용 시
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
        
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{config.TRAIN.EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({"train/loss": round(loss.item(),4)})
                
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # 첫 번째 학습률을 가져옵니다.
        print(f'Epoch [{epoch+1}/{config.TRAIN.EPOCHS}] | Learning Rate: {current_lr}') 

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % config.TRAIN.VAL_EVERY == 0:
            dice, class_dices, val_loss = validation(epoch + 1, model, val_loader, criterion)
            wandb.log({"val/avg_dice": dice, **class_dices, "val/loss": val_loss})
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {config.MODEL.SAVED_DIR}")
                best_dice = dice
                save_model(config, model)
                wandb_model_log(config)
    wandb.finish()
    
def main():
    tf_train = DataTransforms.get_transforms("train")
    tf_valid = DataTransforms.get_transforms("valid")
    
    train_dataset = XRayDataset(is_train=True, transforms=tf_train, config=config)
    valid_dataset = XRayDataset(is_train=False, transforms=tf_valid, config=config)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )


    # model 불러오기
    model = UNet()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=config.TRAIN.LR, weight_decay=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=100, 
                                    gamma=0.1)

    # 학습 시작
    set_seed(config)
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == '__main__':
    main()