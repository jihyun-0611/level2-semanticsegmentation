import albumentations as A

class DataTransforms:
    def get_transforms(phase):
        if phase == "train":
            return A.Compose([
                A.Resize(512, 512),
                # 필요한 augmentation 추가
            ])
        elif phase == "valid" or phase == "test":
            return A.Compose([
                A.Resize(512, 512),
            ])