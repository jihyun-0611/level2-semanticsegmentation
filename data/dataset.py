import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from config.config import Config

config = Config('config.yaml')

CLASS2IND = {v: i for i, v in enumerate(config.DATA.CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, use_pickle=True, pickle_dir="/data/ephemeral/home/data/pickle_data"):
        self.is_train = is_train
        self.transforms = transforms
        self.use_pickle = use_pickle
        self.pickle_dir = pickle_dir
        os.makedirs(self.pickle_dir, exist_ok=True)  # npy 저장 폴더 생성
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=config.DATA.IMAGE_ROOT)
            for root, _dirs, files in os.walk(config.DATA.IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=config.DATA.LABEL_ROOT)
            for root, _dirs, files in os.walk(config.DATA.LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

        _filenames = np.array(sorted(pngs))
        _labelnames = np.array(sorted(jsons))
        
        # split train-valid
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for fname in _filenames]
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break

        self.filenames = filenames
        self.labelnames = labelnames

        # .npy 파일이 존재하지 않으면 미리 생성합니다.
        if use_pickle:
            for image_name, label_name in zip(self.filenames, self.labelnames):
                # npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}.npy")
                image_npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}_image.npz")
                label_npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}_label.npz")
                if not os.path.exists(image_npy_path):
                    data = self._create_data(image_name, label_name)
                    np.savez_compressed(image_npy_path, data[0]) 
                    np.savez_compressed(label_npy_path, data[1])  # npy 파일로 저장
    
    def _create_data(self, image_name, label_name):
        """이미지와 레이블 데이터를 생성해 반환합니다."""
        image_path = os.path.join(config.DATA.IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        image = image / 255.0

        label_path = os.path.join(config.DATA.LABEL_ROOT, label_name)
        label_shape = tuple(image.shape[:2]) + (len(config.DATA.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
            
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        return image, label

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]

        # npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}.npy")
        image_npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}_image.npz")
        label_npy_path = os.path.join(self.pickle_dir, f"{os.path.basename(image_name)}_label.npz")

        # .npy에서 이미지와 레이블 불러오기
        image = np.load(image_npy_path, allow_pickle=True)['arr_0']
        label = np.load(label_npy_path, allow_pickle=True)['arr_0']

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # channel first 포맷으로 변경
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=config.DATA.TEST_IMAGE_ROOT)
            for root, _dirs, files in os.walk(config.DATA.TEST_IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(config.DATA.TEST_IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name