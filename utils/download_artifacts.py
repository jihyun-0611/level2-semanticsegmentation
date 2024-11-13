import os.path as osp
from typing import Tuple
import wandb
from config.config import Config

config = Config('config.yaml')

def download_artifacts(
    project: str,
    entity: str,
    run_id: str,
    model_filename: str,
) -> Tuple[str, str]:
    """
    WandB에서 모델과 데이터셋 아티팩트 다운로드
    
    model_filename: 다운로드 하고 싶은 모델 파일 이름

    Returns:
        model_path: 다운로드된 모델 파일 경로
    """
    model_artifact_name=f"{entity}/{project}/{model_filename}:latest"
    
    # wandb 초기화
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must"
    )
    
    try:
        # 모델 아티팩트 다운로드
        model_artifact = wandb.use_artifact(model_artifact_name, type='model')
        model_dir = model_artifact.download()
        model_path = osp.join(model_dir, model_filename)
        
        return model_path
        
    finally:
        # wandb 종료
        wandb.finish()
        
model_path = download_artifacts(
    project="semantic-segmentation",
    entity="lv2-ss-",
    run_id="bcm6tz5j", # 다운로드 하고 싶은 run_id 작성해주셔야 합니다. wandb UI에서 overview 탭에서 확인할 수 있어요. 
    model_filename=config.MODEL.MODEL_NAME, 
)

# 다운로드된 경로 출력
print(f"모델 경로: {model_path}")