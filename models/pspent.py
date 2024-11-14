import segmentation_models_pytorch as smp
from models.base_model import BaseModel

class pspent(BaseModel):
    def __init__(self, config):
        super(pspent, self).__init__()
        params = config.MODEL.PARAMS
        self.model = smp.PSPNet(
            encoder_name=params.BACKBONE,
            encoder_weights="imagenet",
            in_channels=params.IN_CHANNELS,
            classes=params.OUT_CHANNELS
        )

    def forward(self, image):
        return self.model(image)
    
    def get_model(self):
        return self.model