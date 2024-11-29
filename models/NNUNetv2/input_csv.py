import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.metrics import encode_mask_to_rle

# 클래스 매핑 정의
LABEL_MAPPING = {
    "background": 0, "finger-1": 1, "finger-2": 2, "finger-3": 3, "finger-4": 4, "finger-5": 5,
    "finger-6": 6, "finger-7": 7, "finger-8": 8, "finger-9": 9, "finger-10": 10,
    "finger-11": 11, "finger-12": 12, "finger-13": 13, "finger-14": 14, "finger-15": 15,
    "finger-16": 16, "finger-17": 17, "finger-18": 18, "finger-19": 19,
    "Trapezium": 20, "Trapezoid": 21, "Capitate": 22, "Hamate": 23,
    "Scaphoid": 24, "Lunate": 25, "Triquetrum": 26, "Pisiform": 27, "Radius": 28,
    "Ulna": 29, 
}

def nifti_to_csv(nifti_dir, output_path):
    results = []

    files = sorted(
        [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')],
        key=lambda x: int(x.split('_')[0].replace('ID', ''))  # ID040 -> 40로 변환하여 정렬
    )
    
    for filename in tqdm(files):
        if not filename.endswith('.nii.gz'):
            continue
            
        nifti_path = os.path.join(nifti_dir, filename)
        img = nib.load(nifti_path)
        mask_data = img.get_fdata()
        
        # ID 부분을 제거하고 파일 이름만 추출
        image_name = filename.replace('.nii.gz', '').split('_')[1] + '.png' # ID### 부분을 제거
        
        for class_name, class_id in LABEL_MAPPING.items():
            if class_name == "background":
                continue
                
            class_mask = (mask_data == class_id).astype(np.uint8)
            rle = encode_mask_to_rle(class_mask)
            
            results.append({
                "image_name": image_name,
                "class": class_name,
                "rle": rle
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # 입력 디렉토리와 출력 경로 설정
    nifti_dir = "/data/ephemeral/home/nnUNet-master/OUTPUT_FOLDER"  # nii.gz 파일이 있는 디렉토리
    output_path = "output.csv"  # 저장할 CSV 파일 경로
    
    nifti_to_csv(nifti_dir, output_path)