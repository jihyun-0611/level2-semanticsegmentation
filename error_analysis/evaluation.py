import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os

from config.config import Config
from data.dataset import XRayDataset
from utils.metrics import encode_mask_to_rle, dice_coef
from data.augmentation import DataTransforms
import models 

import segmentation_models_pytorch as smp
import argparse

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def test(model, data_loader, thr=0.5):
    model.eval()

    dices = []
    rles = []
    classes = []
    filenames = []
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
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
            
            ###############################################################################################
            # save validation outputs in list
            dataset = data_loader.dataset
            batch_filenames = [dataset.filenames[i] for i in range(step * data_loader.batch_size, (step + 1) * data_loader.batch_size)]
            
            for output, image_name in zip(outputs, batch_filenames):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm.cpu())
                    rles.append(rle)
                    classes.append(data_loader.dataset.IND2CLASS[c])
                    filenames.append(image_name)
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(config.DATA.CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    print(f'average dice : {avg_dice}')
                    
    image_name = [os.path.basename(f) for f in filenames]
    os.makedirs(config.TRAIN.OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        config.TRAIN.OUTPUT_DIR,
        config.TRAIN.CSV_NAME
    )

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(output_path, index=False)
    print(f"Validation Results saved to {output_path}")
             
    return rles, filenames

def main(config):
    model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
    params = config.MODEL.PARAMS
    model = model_class(config)
    model = smp.Unet(
        encoder_name=params.BACKBONE, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,                     # model output channels (number of classes in your dataset)
    )
    model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    model.load_state_dict(torch.load(model_path), strict=False)
    # model =torch.load(os.path.join(config.MODEL.SAVED_DIR, 'artifacts',config.MODEL.MODEL_NAME))
    tf = DataTransforms.get_transforms("valid")

    valid_dataset = XRayDataset(is_train=False, transforms=tf, config=config)
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    test(model, valid_loader)

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config)