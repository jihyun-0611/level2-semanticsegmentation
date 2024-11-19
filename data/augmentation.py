import albumentations as A
import yaml

class DataTransforms:
    def __init__(self, config):
        self.config = config

    def get_transforms(self, phase):
        transforms_list = []
        
        if phase == "train":
            transforms = self.config.TRAIN.TRANSFORMS
        elif phase == "valid" or phase == "test":
            transforms = self.config.INFERENCE.TRANSFORMS
        
        for transform in transforms:
            transform_name = transform['NAME']
            params = transform['PARAMS']

            try:
                transform_class = getattr(A, transform_name)
                transforms_list.append(transform_class(**params))
            except AttributeError:
                raise ValueError(f"Albumentations 라이브러리에 {transform_name} 라는 증강 기법이 없습니다. 증강 기법명을 확인하세요.")
        
        return A.Compose(transforms_list)