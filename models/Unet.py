import segmentation_models_pytorch as smp
from models.base_model import BaseModel

class UNet(BaseModel):
    def __init__(self):
        super(UNet, self).__init__()
        self.model = smp.Unet(encoder_name="efficientnet-b0", 
                              encoder_weights=None, 
                              in_channels=3, 
                              classes=29)

    def forward(self, image):
        return self.model(image)
    
    def get_model(self):
        return self.model