import os
import yaml
from easydict import EasyDict

class Config:
    def __init__(self, config_path, default_config=None):
        # 기본 설정값 초기화
        self.default_config = EasyDict(default_config) if default_config else EasyDict()

        # 사용자 정의 설정 로드
        with open(config_path, 'r') as f:
            user_config = EasyDict(yaml.safe_load(f))
        
        # 기본 설정값과 병합
        self.config = self.merge_config(user_config, self.default_config)
        self.__dict__.update(self.config)

    @staticmethod
    def merge_config(user_config, default_config):
        """
        기본 설정(default_config)과 사용자 설정(user_config)을 병합.
        """
        for key, value in default_config.items():
            if key not in user_config:
                user_config[key] = value
            elif isinstance(value, dict):
                user_config[key] = Config.merge_config(user_config[key], value)
        return user_config