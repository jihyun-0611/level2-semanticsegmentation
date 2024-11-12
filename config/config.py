import os
import yaml
from easydict import EasyDict

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = EasyDict(yaml.safe_load(f))
            self.__dict__.update(self.config)
