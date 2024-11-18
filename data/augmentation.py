import albumentations as A

class DataTransforms:
    def get_transforms(phase):
        if phase == "train":
            return A.Compose([
                A.Resize(1024, 1024),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # 필요한 augmentation 추가
            ])
        elif phase == "valid" or phase == "test":
            return A.Compose([
                A.Resize(1024, 1024),
            ])