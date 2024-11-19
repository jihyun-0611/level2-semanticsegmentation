# 개요

이 디렉토리에는 다양한 실험 설정을 정의한 YAML 파일들이 포함되어 있습니다.

> **⚠️ 주의:**  
> 반복 실험 시 `experiments` 디렉토리 내 **모든 YAML 파일**이 실행됩니다.  
> 완료된 실험 파일은 반드시 `experiments/completed` 폴더로 이동해 주세요.

## YAML 파일 작성 규칙

- **파일 이름 형식**: `실험번호_내용.yaml` 형식을 사용해 주세요.
- **업로드 지침**: 실험이 완료된 후, 다른 팀원이 참고 및 구현할 수 있도록 YAML 파일을 GitHub에 업로드해 주세요. 

    (⚠️ 다시 말씀드리지만, 반드시 `experiments/completed` 폴더에 넣어주세요.)
- **LOSS, SCHEDULER, OPTIMIZER 관련** : LOSS, SCHEDULER, OPTIMIZER의 파라미터는 PARAM 인자 밑에 작성해주셔야 합니다. 그렇지 않으면 train.py의 DEFAULT_TRAIN_CONFIG의 파라미터로 실행될 수 있습니다.

## 실험 실행 방법

### 단일 실험 실행

```bash
bash train.sh
```


### 반복 실험
```bash
bash multi_train.sh
```