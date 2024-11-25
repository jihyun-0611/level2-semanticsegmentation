import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os

from config.config import Config
from data.dataset import XRayInferenceDataset
from utils.metrics import encode_mask_to_rle
from data.augmentation import DataTransforms
import models 

import argparse

from utils.utils import save_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    result_rles = {'image_names': [], 'classes': [], 'rles': []}

    with torch.no_grad():

        patch_outputs = []

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
            # outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            patch_outputs.append(outputs)

            if (step + 1) % 3 == 0:
                dataset = data_loader.dataset
                image_name = dataset.filenames[step * 3]
                full_output = torch.zeros((outputs.size(1), 2048, 2048)).cuda()

                for idx, (patch_output) in enumerate(torch.cat(patch_outputs)):
                    y = (idx // 3) *  (512)
                    x = (idx % 3) * (512)
                    full_output[:, y:y + 1024, x:x + 1024] += patch_output

                div = torch.tensor([[1, 2, 2, 1],
                                    [2, 4, 4, 2],
                                    [2, 4, 4, 2],
                                    [1, 2, 2, 1]]).float().cuda()
                div_h, div_w = full_output.size(-2) // 4, full_output.size(-1) // 4
                div_expanded = div.repeat_interleave(div_h, dim=0).repeat_interleave(div_w, dim=1)
                full_output = full_output / div_expanded
                full_output = (full_output > thr)

                for c, segm in enumerate(full_output):
                    rle = encode_mask_to_rle(segm.cpu())
                    result_rles['rles'].append(rle)
                    result_rles['classes'].append(dataset.IND2CLASS[c])
                    result_rles['image_names'].append(os.path.basename(image_name))

                patch_outputs = []
                    
    return result_rles

def main(config):
    model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
    # try:
    #     model = model_class(config).get_model()
    #     model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    #     model.load_state_dict(torch.load(model_path))
    # except Exception as e:
    #     model = model_class(config)
    #     model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    #     model.load_state_dict(torch.load(model_path))
    #     print(f"에러로 인해 get_model()을 사용하여 모델을 로드합니다.")
    
    model = model_class(config).get_model() # 에러나면 get_model 이거 빼세요
    model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    print("model path!!!!!!!!!!!!!!!!!!!!")
    print(model_path)
    state_dict = torch.load(model_path)
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    state_dict = {key: value for key, value in state_dict.items() if 'n_averaged' not in key}
    model.load_state_dict(state_dict)



    data_transforms = DataTransforms(config)
    tf = data_transforms.get_transforms("valid")

    test_dataset = XRayInferenceDataset(transforms=tf, config=config)
    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=3, # 배치 사이즈 3으로 고정 메모리 문제 시 1로 변경 후 44, 46 라인 3을 9로 변경
    shuffle=False,
    num_workers=0,
    drop_last=False
    )

    rles = test(model, test_loader)
    save_csv(config, rles, mode='INFERENCE')


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config)