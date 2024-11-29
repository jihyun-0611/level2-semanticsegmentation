from models.base_model import BaseModel
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR

class SwinUNETR(BaseModel):
    def __init__(self, config):
        super().__init__()
        params = config.MODEL.PARAMS
        self.model = MonaiSwinUNETR(
            img_size=(params.HEIGHT, params.WIDTH),
            in_channels=params.IN_CHANNELS,
            out_channels=params.OUT_CHANNELS,
            feature_size=params.FEATURE_SIZE,
            use_checkpoint=params.USE_CHECKPOINT,
            spatial_dims=2,
            drop_rate=params.DROP_RATE,
            attn_drop_rate=params.ATTN_DROP_RATE,
            dropout_path_rate=params.DROPOUT_PATH_RATE
        )
        
    def forward(self, image):
        return self.model(image)

    def get_model(self):
        return self.model