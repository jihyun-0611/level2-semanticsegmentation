from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import os
import json
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm
import numpy as np
import cv2

# 클래스와 매핑된 번호 정의
LABEL_MAPPING = {
    "background": 0, "finger-1": 1, "finger-2": 2, "finger-3": 3, "finger-4": 4, "finger-5": 5,
    "finger-6": 6, "finger-7": 7, "finger-8": 8, "finger-9": 9, "finger-10": 10,
    "finger-11": 11, "finger-12": 12, "finger-13": 13, "finger-14": 14, "finger-15": 15,
    "finger-16": 16, "finger-17": 17, "finger-18": 18, "finger-19": 19,
    "Trapezium": 20, "Trapezoid": 21, "Capitate": 22, "Hamate": 23,
    "Scaphoid": 24, "Lunate": 25, "Triquetrum": 26, "Pisiform": 27, "Radius": 28,
    "Ulna": 29, "Trapezium_Trapezoid_Overlap": 30,
    "Triquetrum_Pisiform_Overlap": 31
}

if __name__ == '__main__':
    # 입력 데이터 및 출력 경로 설정
    input_train_image_dir = "/data/ephemeral/home/data/train/DCM"
    input_train_annotation_dir = "/data/ephemeral/home/data/train/outputs_json"
    input_test_image_dir = "/data/ephemeral/home/data/test/DCM"
    target_dataset_id = 4
    target_dataset_name = f"Dataset{target_dataset_id:03.0f}_HandSegmentation"
    output_base = nnUNet_raw
    output_dir = os.path.join(output_base, target_dataset_name)

    # nnUNet 데이터 구조 생성
    maybe_mkdir_p(output_dir)
    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")  # Test 이미지 디렉토리
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)
    
    # 데이터 변환 함수 정의
    def process_data(input_image_dir, input_annotation_dir, output_image_dir, output_label_dir=None, is_test=False):
        cases = sorted(os.listdir(input_image_dir))
        for case_id in tqdm(cases, desc="Processing cases"):
            case_image_dir = os.path.join(input_image_dir, case_id)
            case_annotation_dir = (
                os.path.join(input_annotation_dir, case_id) if not is_test else None
            )

            if not os.path.isdir(case_image_dir):
                continue

            for image_file in os.listdir(case_image_dir):
                if not image_file.endswith(".png"):
                    continue

                image_path = os.path.join(case_image_dir, image_file)
                output_image_path = os.path.join(output_image_dir, f"{case_id}_{image_file.replace('.png', '_0000.nii.gz')}")

                # 이미지 저장
                image = cv2.imread(image_path) # (2048, 2048, 3)
                image = image / 255.0
                image_sitk = sitk.GetImageFromArray(image) # (3, 2048, 2048)

                sitk.WriteImage(image_sitk, output_image_path)

                if is_test or not case_annotation_dir:
                    continue

                annotation_file = image_file.replace(".png", ".json")
                annotation_path = os.path.join(case_annotation_dir, annotation_file)

                if not os.path.exists(annotation_path):
                    print(f"Annotation not found for {image_file}")
                    continue

                output_label_path = os.path.join(output_label_dir, f"{case_id}_{image_file.replace('.png', '.nii.gz')}")

                # 라벨 마스크 생성
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)["annotations"]

                    # image.shape[:2] = (2048, 2048)
                    label_shape = image.shape[:2] + (len(LABEL_MAPPING),)  # (2048, 2048, 30)
                    
                    label_array = np.zeros(label_shape, dtype=np.uint8)
                
                    for annotation in annotations:
                        points = np.array(annotation["points"])

                        label_name = annotation["label"]

                        if label_name not in LABEL_MAPPING:
                            print(f"Warning: Undefined label '{label_name}' in {annotation_path}")
                            continue

                        label_value = LABEL_MAPPING[label_name]
                        label = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(label, [points], 1)
                        label_array[..., label_value] = label

                # 라벨 저장
                label_image = sitk.GetImageFromArray(label_array) # label_array : (30, 2048, 2048)
                sitk.WriteImage(label_image, output_label_path)

    # Train 데이터 처리
    process_data(input_train_image_dir, input_train_annotation_dir, imagesTr, labelsTr, is_test=False)

    # Test 데이터 처리
    process_data(input_test_image_dir, None, imagesTs, is_test=True)

    # dataset.json 생성
    generate_dataset_json(
        output_dir,
        {0: "X-ray"},
        LABEL_MAPPING,
        800,
        ".nii.gz",
        None,
        target_dataset_name,
        overwrite_image_reader_writer="SimpleITKIO",
        reference= "https://boostcamp.connect.or.kr/",
        licence= "Naver_AI_boostcamp",
    )
    print("Data preprocessing complete!")
