import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
#import torchvision.models as models
from tqdm import tqdm
import os

from utils.utils import set_seed, save_model, wandb_model_log, save_csv
from utils.metrics import dice_coef, encode_mask_to_rle
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.loss import get_loss
from data.dataset import XRayDataset
from data.augmentation import DataTransforms
import models

from config.config import Config
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    result_rles = {'image_names': [], 'classes': [], 'rles': []}
    
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        patch_outputs = []
        patch_masks = []

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # if step == 30:    # 몇개의 이미지로 동작을 확인하고 돌리는 것을 추천
            #     break

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

            
            patch_outputs.append(outputs)
            patch_masks.append(masks)


            if (step + 1) % 3 == 0:
                dataset = data_loader.dataset
                image_name = dataset.filenames[step * 3]

                full_output = torch.zeros((outputs.size(1), mask_h * 2, mask_w * 2)).cuda()
                full_mask = torch.zeros((masks.size(1), mask_h * 2, mask_w * 2)).cuda()

                for idx, (patch_output, patch_mask) in enumerate(zip(torch.cat(patch_outputs), torch.cat(patch_masks))):
                    y = (idx // 3) *  (mask_h // 2)
                    x = (idx % 3) * (mask_w // 2)
                    full_output[:, y:y + mask_h, x:x + mask_w] += patch_output
                    full_mask[:, y:y + mask_h, x:x + mask_w] += patch_mask
                
                div = torch.tensor([[1, 2, 2, 1],
                                    [2, 4, 4, 2],
                                    [2, 4, 4, 2],
                                    [1, 2, 2, 1]]).float().cuda()
                div_h, div_w = full_mask.size(-2) // 4, full_mask.size(-1) // 4
                div_expanded = div.repeat_interleave(div_h, dim=0).repeat_interleave(div_w, dim=1)
                full_output = full_output / div_expanded

                full_output = (full_output > thr)
                full_mask = (full_mask > 0)

                dice = dice_coef(full_output.unsqueeze(0), full_mask.unsqueeze(0))
                dices.append(dice)

                for c, segm in enumerate(full_output):
                    rle = encode_mask_to_rle(segm.cpu())
                    result_rles['rles'].append(rle)
                    result_rles['classes'].append(dataset.IND2CLASS[c])
                    result_rles['image_names'].append(os.path.basename(image_name))

                patch_outputs = []
                patch_masks = []
            
        print('val total loss: ', (total_loss/cnt))
                
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
    print('avg_dice: ', avg_dice)
    
    return avg_dice, dice_dict, result_rles
    
def main():
    data_transforms = DataTransforms(config)
    tf_valid = data_transforms.get_transforms("valid")
    valid_dataset = XRayDataset(is_train=False, transforms=tf_valid, config=config)
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=3, # 배치사이즈를 반드시 3으로 고정, 메모리 문제 발생 시 1로 수정하고 71,73라인의 3을 9로 변경
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # model 불러오기
    model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
    model = model_class(config).get_model()
    
    model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME) # 에러나면 get_model() 이거 빼세요
    print(model_path)
    state_dict = torch.load(model_path)
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    state_dict = {key: value for key, value in state_dict.items() if 'n_averaged' not in key}
    model.load_state_dict(state_dict)

    # 학습 시작
    set_seed(config)

    criterion = nn.BCEWithLogitsLoss()
    dice, class_dices, rles = validation(1, model, valid_loader, criterion)
    save_csv(config, rles, mode='TRAIN', epoch=1)

if __name__ == '__main__':
    args = parse_args()
    config = Config(args.config)
    main()
