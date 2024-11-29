from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import numpy as np
import os
from sklearn.model_selection import GroupKFold


@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, mode, **kwargs):
        self.classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
            ]
        self.mode = mode
        DATA_DIR = f'/data/ephemeral/home/data/{mode}'
        self.image_root = os.path.join(DATA_DIR, 'DCM')
        if self.mode == 'train':
            self.label_root = os.path.join(DATA_DIR, 'outputs_json') 
        else:
            self.label_root = None
        self.CLASS2IND = {v: i for i, v in enumerate(self.classes)}
        self.IND2CLASS = {v: k for k, v in self.CLASS2IND.items()}

        # os.makedirs(self.pickle_dir, exist_ok=True)  # npy 저장 폴더 생성
        
        super().__init__(**kwargs)
        
    def load_data_list(self):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = np.array(sorted(pngs))
        
        if self.mode == 'train':
            jsons = {
                os.path.relpath(os.path.join(root, fname), start=self.label_root)
                for root, _dirs, files in os.walk(self.label_root)
                for fname in files
                if os.path.splitext(fname)[1].lower() == ".json"
            }
            _labelnames = np.array(sorted(jsons))
        
            # split train-valid
            groups = [os.path.dirname(fname) for fname in _filenames]
            ys = [0 for fname in _filenames]
            gkf = GroupKFold(n_splits=5)
            
            filenames = []
            labelnames = []
            for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
                if self.mode == 'train':
                    # 0번을 validation dataset으로 사용합니다.
                    if i == 3:
                        continue
                        
                    filenames += list(_filenames[y])
                    labelnames += list(_labelnames[y])
                
                else:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    
                    # skip i > 0
                    break

            data_list = []
            for (img_path, ann_path) in zip(filenames, labelnames):
                data_info = dict(
                    img_path = os.path.join(self.image_root, img_path),
                    seg_map_path = os.path.join(self.label_root, ann_path),
                )
                data_list.append(data_info)
             
        # mode == 'test'
        else:
            data_list = []
            for img_path in _filenames:
                data_info = dict(
                    img_path = os.path.join(self.image_root, img_path)
                )
                data_list.append(data_info)
        
        return data_list
