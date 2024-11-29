import os
import json
import shutil
import zipfile
from tempfile import TemporaryDirectory
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 원본 데이터 root
    parser.add_argument('--IMAGE_ROOT', type=str, default="/data/ephemeral/home/data/train/DCM")
    parser.add_argument('--LABEL_ROOT', type=str, default="/data/ephemeral/home/data/train/outputs_json")
    
    parser.add_argument('--convert', type=str, choices=['to_coco', 'from_coco'], required=True)
    parser.add_argument('--output_folder', type=str, help="Output folder path")
    parser.add_argument('--coco_path', type=str, help="Only required for from_coco") 
    return parser.parse_args()


# Orginal format -> COCO format
def convert_to_coco(output_folder):
    """
    output_folder/train_coco.json : coco format으로 변환한 annotation 파일
    output_folder/images.zip : annotation 변환한 이미지 파일들을 압축한 zip 파일
    """

    jsons = sorted([
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    ])

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    with TemporaryDirectory() as temp_image_dir:
        # images, annotations
        anno_n = 0
        for i, j in enumerate(jsons):
            image_name = f"{j[:-5]}.png"
            image_path = os.path.join(IMAGE_ROOT, image_name)
            save_name = image_name.replace('/', '_')
            temp_image_path = os.path.join(temp_image_dir, save_name)
            shutil.copy(image_path, temp_image_path)

            label_path = os.path.join(LABEL_ROOT, j)
            with open(label_path, 'r') as f:
                annotation = json.load(f)

            coco_format["images"].append({
                "id": i + 1,
                "width": annotation['metadata']['width'],
                "height": annotation['metadata']['height'],
                "file_name": save_name,
                "license": 0,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": 0
            })
            for anno in annotation['annotations']:
                anno_n += 1
                coco_format["annotations"].append({
                    "id": anno_n,
                    "image_id": i + 1,
                    "category_id": CLASS2IND[anno['label']] + 1,
                    "segmentation": [[p for point in anno['points'] for p in point]],
                    "area": 0,
                    "iscrowd": 0
                })

        # categories
        for i in range(len(CLASSES)):
            coco_format["categories"].append({
                "id": i + 1,
                "name": CLASSES[i],
                "supercategory": CLASSES[i]
            })

        # COCO JSON 저장
        os.makedirs(output_folder, exist_ok=True)
        json_output_path = os.path.join(output_folder, "train_coco.json")
        with open(json_output_path, 'w') as out:
            json.dump(coco_format, out, indent=4)

        # 이미지를 ZIP 파일로 압축
        zip_output_base = os.path.join(output_folder, "images")
        shutil.make_archive(zip_output_base, 'zip', temp_image_dir)

    print(f"COCO format saved to {output_folder}")


# COCO format -> Original format
def convert_from_coco(coco_path, output_folder):
    """
    output_folder/train/outputs_json/ID000 : original data와 마찬가지로 ID폴더 아래 새로운 json 파일 생성
    output_folder/train/DCM/ID000 : coco json파일에 포함된 image만 ID폴더 아래 저장
    """
    new_image_root = os.path.join(output_folder, 'train', 'DCM')
    new_label_root = os.path.join(output_folder, 'train', 'outputs_json')
    os.makedirs(new_image_root, exist_ok=True)
    os.makedirs(new_label_root, exist_ok=True)

    with open(coco_path, 'r') as f:
        coco = json.load(f) 

    categories = dict()
    for c in coco["categories"]:
        categories[c['id']] = c['name']

    for i, image_info in enumerate(coco["images"]):
        image_id = image_info['id']
        folder_id = image_info['file_name'][:5]
        filename = image_info['file_name'][6:]
        annotation = coco['annotations'][i*29 : i*29+29]

        original_image_path = os.path.join(IMAGE_ROOT, folder_id, filename)

        image_output_folder = os.path.join(new_image_root, folder_id)
        os.makedirs(image_output_folder, exist_ok=True)
        new_image_path = os.path.join(image_output_folder, filename)
        shutil.copy(original_image_path, new_image_path)
        
        original_json_path = os.path.join(LABEL_ROOT, folder_id, f'{filename[:-4]}.json')
        with open(original_json_path, 'r') as origin:
            original_json = json.load(origin)
        
        for anno in annotation:
            if anno['image_id'] != image_id:
                raise ValueError(f"Missing class -> id {image_id} image")
            category = categories[anno['category_id']]

            if len(anno['segmentation']) != 1:
                raise ValueError(f"More than 1 label in a class -> id {image_id} image")
            points = np.reshape(np.array(anno['segmentation'][0]).astype(np.int32), (-1, 2)).tolist()

            for j, original_anno in enumerate(original_json["annotations"]):
                if original_anno['label'] == category:
                    original_json["annotations"][j]['points'] = points
                    break
        
        label_output_folder = os.path.join(new_label_root, folder_id)
        os.makedirs(label_output_folder, exist_ok=True)
        new_label_path = os.path.join(label_output_folder, f'{filename[:-4]}.json')
        with open(new_label_path, 'w') as out:
            json.dump(original_json, out, indent=4)

    print(f"Original format saved to {output_folder}")


if __name__ == "__main__":

    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

    args = parse_args()
    IMAGE_ROOT = args.IMAGE_ROOT
    LABEL_ROOT = args.LABEL_ROOT

    if args.convert == "to_coco":
        if not args.output_folder:
            args.output_folder = "/data/ephemeral/home/data_coco"
        convert_to_coco(output_folder=args.output_folder)

    elif args.convert == "from_coco":
        if not args.coco_path:
            raise ValueError("--coco_path is required for 'from_coco'")
        if not args.output_folder:
            args.output_folder = "/data/ephemeral/home/new_data"
        convert_from_coco(args.coco_path, output_folder=args.output_folder)
    else:
        print("Select one: to_coco or from_coco")