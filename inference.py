import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os

from config.config import Config
from data.dataset import XRayInferenceDataset, IND2CLASS
from utils.metrics import encode_mask_to_rle
from data.augmentation import DataTransforms

config = Config('config.yaml')

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def main():
    model = torch.load(os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME))
    
    tf = DataTransforms.get_transforms("valid")

    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )

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
    main()