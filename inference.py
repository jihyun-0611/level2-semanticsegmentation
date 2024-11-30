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

    rles = []
    filename_and_class = []
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
                    rles.append(rle)
                    filename_and_class.append(f"{data_loader.dataset.IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def sliding_test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    result_rles = {'image_names': [], 'classes': [], 'rles': []}

    with torch.no_grad():

        patch_outputs = []

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
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
    if config.INFERENCE.INFERENCE_SWA:
        try:
            model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
            model = model_class(config)
            model_path=os.path.join(config.MODEL.SAVED_DIR, config.TRAIN.SWA.MODEL_NAME)

            state_dict = torch.load(model_path)
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            state_dict = {key: value for key, value in state_dict.items() if 'n_averaged' not in key}
            model.load_state_dict(state_dict)
        except Exception as e:
            model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
            model = model_class(config).get_model()
            model_path=os.path.join(config.MODEL.SAVED_DIR, config.TRAIN.SWA.MODEL_NAME)

            state_dict = torch.load(model_path)
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            state_dict = {key: value for key, value in state_dict.items() if 'n_averaged' not in key}
            model.load_state_dict(state_dict)
    else:
        try:
            model = model_class(config).get_model()
            model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            model = model_class(config)
            model_path=os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
            model.load_state_dict(torch.load(model_path))
            print(f"에러로 인해 get_model()을 사용하여 모델을 로드합니다.")
    
    data_transforms = DataTransforms(config)
    tf = data_transforms.get_transforms("valid")
    
    batch_size = 3 if config.TRAIN.SLIDING_WINDOW else 2
    test_dataset = XRayInferenceDataset(transforms=tf, config=config)
    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )

    if config.TRAIN.SLIDING_WINDOW:
        rles = test(model, test_loader)
        save_csv(config, rles, mode='INFERENCE')
    else:
        rles, filename_and_class = test(model, test_loader)

        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]

        os.makedirs(config.INFERENCE.OUTPUT_DIR, exist_ok=True)
        
        output_path = os.path.join(
            config.INFERENCE.OUTPUT_DIR,
            config.INFERENCE.CSV_NAME
        )

        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })

        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config)