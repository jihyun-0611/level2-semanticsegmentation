import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os

from config.config import Config
from data.dataset import XRayInferenceDataset
from utils.utils import save_csv
from utils.metrics import encode_mask_to_rle
from data.augmentation import DataTransforms
import models 

import segmentation_models_pytorch as smp
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    result_rles = {'image_names': [], 'classes': [], 'rles': []}
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    result_rles['rles'].append(rle)
                    result_rles['classes'].append(data_loader.dataset.IND2CLASS[c])
                    result_rles['image_names'].append(os.path.basename(image_name))
                    
    return result_rles

def main(config):
    model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
    try:
        model = model_class(config).get_model()
        model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        model = model_class(config)
        model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
        model.load_state_dict(torch.load(model_path))
        print(f"에러로 인해 get_model()을 사용하여 모델을 로드합니다.")
    
    tf = DataTransforms.get_transforms("valid")

    test_dataset = XRayInferenceDataset(transforms=tf, config=config)
    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )

    rles = test(model, test_loader)
    save_csv(config, rles, mode='INFERENCE', epoch=None)

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config)