# 🦴 Hand Bone Image Segmentation

<p align="center">
    </picture>
    <div align="center">
        <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
        <img src="https://img.shields.io/badge/W&B-FFBE00.svg?style=for-the-badge&logo=weightsandbiases&logoColor=white">
        <img src="https://img.shields.io/badge/tmux-1BB91F?style=for-the-badge&logo=tmux&logoColor=white">
        <img src="https://img.shields.io/badge/yaml-CB171E?style=for-the-badge&logo=yaml&logoColor=black">
    </div>
    </picture>
    <div align="center">
        <img src="https://github.com/user-attachments/assets/38a94eec-738f-4642-8873-729c798d6884" width="300"/>
        <p> (이미지 출처 : <a href="https://en.m.wikipedia.org/wiki/File:X-ray_of_normal_hand_by_dorsoplantar_projection.jpg" target="_blank">Wikipedia</a>)</p>
    </div>
</p>

<br />

## ✏️ Introduction
Bone Image Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 골절 진단, 성장 평가, 수술 계획 등 다양한 의료 분야에 활용될 수 있습니다. 해당 대회는 제공된 X-ray 손 뼈 이미지에서 29개의 뼈를 분할하는 AI 모델을 개발하는 것을 목표로 합니다. 대회의 성능은 Semantic Segmentation task에서 대표적으로 사용되는 dice coefficient로 평가됩니다.

<br />

## 📅 Schedule
프로젝트 전체 일정

- 2024.11.13 ~ 2024.11.28

프로젝트 세부일정

<img src="https://github.com/user-attachments/assets/7d17ba49-be2f-4bd2-8a08-42f7f0302ddc" width="500"/>

## 🕵️ 프로젝트 파이프라인 

<img src="https://github.com/user-attachments/assets/5300dad3-8e0f-4927-ade9-241b01771e6d" width="500"/>

각 파이프라인에 대한 상세한 내용은 아래 링크를 통해 확인할 수 있습니다.

- [MLFlow 및 Wandb 연동](https://shadowed-fact-f9b.notion.site/Wandb-with-mmdection-train-8854fc9596a743ebb7ecdbb894dbd807?pvs=4)
- [데이터 EDA 및 Streamlit 시각화](https://shadowed-fact-f9b.notion.site/EDA-Streamlit-bd10bb80c7704431b27c05929899bc4e?pvs=4)
- [Validation 전략 구축](https://shadowed-fact-f9b.notion.site/Validation-d56cc4f852334249905ef1c99b05133d?pvs=4)
- [모델 실험 및 평가](https://shadowed-fact-f9b.notion.site/4287a4ea70f145739bf45738ae35051d?pvs=4)
- [모델 앙상블 실험](https://shadowed-fact-f9b.notion.site/ensemble-ca0522e34a544108a8f2b1ff66ca7ed3?pvs=4)

<br />

## 🥈 Result
Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.
<img align="center" src="https://github.com/user-attachments/assets/da9d8fbc-7b11-4c82-a377-4a90d7041968" width="600" height="70">

<br />

## 🗃️ Dataset Structure
```
dataset/
├── meta_data.xlsx
├── test
│   └── DCM
│       ├── ID040
│       │   ├── image1661319116107.png
│       │   └── image1661319145363.png
│       └── ID041
│           ├── image1661319356239.png
│           └── image1661319390106.png
└── train
    ├── DCM
    │   ├── ID001
    │   │   ├── image1661130828152_R.png
    │   │   └── image1661130891365_L.png
    │   └── ID002
    │       ├── image1661144206667.png
    │       └── image1661144246917.png
    └── outputs_json
        ├── ID001
        │   ├── image1661130828152_R.json
        │   └── image1661130891365_L.json
        └── ID002
            ├── image1661144206667.json
            └── image1661144246917.json

```
- 데이터셋은 한 사람의 양 손을 촬영한 X-Ray 이미지이며, 하나의 ID에 한 사람에 대한 오른손 및 왼손의 이미지가 들어있습니다. 각 이미지는 의학적으로 분류된 손 뼈 29가지의 클래스를 가지며, 2048x2048 크기의 train 이미지 800장, test 이미지 288장으로 구성됩니다.
- 또한, 각 ID에 해당하는 사람의 나이, 성별, 체중, 신장에 대한 Mata_data가 xlsx로 주어집니다. 

### Train json

Train json 파일은 각 이미지에 대한 annotations이 포함되며, id, type, attributes, points, label로 구성되어 있습니다.
- Images
  ```json
    {"annotations": [{
        "id": "id", 
        "type": "poly_seg", 
        "attributes": {}, 
        "points": [[[10, 20], [30, 40,] ... ]], 
        "label": "finger-1"} 
        ...]}
  ```
  - ⚠️ 데이터 보안상의 문제로 ID와 Points는 임의로 작성되었습니다.
<br />

## ⚙️ Requirements

### env.
이 프로젝트는 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 훈련 및 테스트되었습니다.

### Installment
이 프로젝트에서는 다양한 라이브러리가 필요합니다. 아래 단계를 따라 필요한 라이브러리를 설치하세요.

#### 1. PyTorch 설치

PyTorch 2.1.0을 설치합니다. 설치 방법은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)를 참고하세요.

#### 2. 프로젝트 의존성 설치

다음 명령어를 실행하여 프로젝트를 클론하고 필요한 의존성을 설치합니다:

```bash
git clone https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3.git
cd level2-objectdetection-cv-23
pip install -r requirements.txt
```


<br />

## 🎉 Project

### 1. Structure
  ```bash
project
├── Detectron2
│   ├── detectron2_inference.py
│   └── detectron2_train.py
├── EDA
│   ├── confusion_matrix_trash.py
│   └── Stramlit
│       ├── arial.ttf
│       ├── EDA_Streamlit.py
│       ├── EDA_Streamlit.sh
│       ├── inference_json
│       │   └── val_split_rand411_pred_latest.json
│       └── validation_json
│           └── val_split_random411.json
├── mmdetection2
│   ├── mmdetection2_inference.py
│   ├── mmdetection2_train.py
│   └── mmdetection2_val.py
├── mmdetection3
│   ├── mmdetectionV3_inference.py
│   ├── mmdetectionV3_train.py
│   └── mmdetectionV3_val.py
├── README.md
├── requirements.txt
└── src
    ├── ensemble.py
    └── make_val_dataset.ipynb
```
### 2. EDA
#### 2-1. Streamlit
Train data 및 inference 결과의 EDA을 위해 Streamlit을 활용했습니다. Streamlit을 통해 EDA를 진행하기 위해 다음을 실행하세요.
```bash
bash EDA_Streamlit.sh
```
실행을 위해 다음의 인자가 필요합니다.

  - **dataset_path** : dataset 경로
  - **font_path** : bbox의 시각화를 위한 font 경로 (우리의 Repository에 있는 arial.ttf을 이용하세요)
  - **inference_path** : inference json 파일 경로
  - **validation_path** : validation json 파일 경로
  
데모 실행을 위해 validation_json, inference_json directory에 데모 json 파일이 있습니다.

#### 2-2. confusion_matrix
Confusion matrix를 시각화하기 위해 confusion_matrix_trash.py 코드를 추가하였습니다.

해당 코드는 validation inference 시 confusion matrix도 함께 출력하기 위한 코드로 직접 실행하지 않고 val.py에서 import해 사용합니다. mmdetectionv2_val.py에서 confusion matrix를 출력하는 코드를 확인하실 수 있습니다.

mmdetectionv2_val.py를 실행하면 추론 결과를 담은 json 파일, confusion_matrix를 위한 pickel파일, confusion_matrix png파일이 함께 저장됩니다.
        
### 3. Train and inference
프로젝트를 위해 mmdetection V2 및 V3, Detectron2를 사용했습니다. 각 라이브러리에 해당하는 directory에 train과 inference를 위한 코드가 있습니다.

해당 코드들을 사용하기 위해 mmdetection 및 Detectron2 라이브러리에 포함된 config 파일이 필요합니다. 밑의 링크들을 통해 config 파일과 그에 필요한 구성 요소들을 clone할 수 있습니다.
  
- [mmdetection](https://github.com/open-mmlab/mmdetection) 
- [Detectron2](https://github.com/facebookresearch/detectron2)

[라이브러리명]_val.py 파일은 Streamlit 시각화를 위해 validation inference 결과에 대한 json 파일을 추출하는 코드입니다. Detectron2의 경우 detectron2_inference.py를 통해 json 파일을 추출할 수 있습니다. 
<br />

### 4. ensemble
앙상블을 사용하기 위해 다음을 실행하세요.
```bash
python ./src/ensemble.py
```

아래 변수 값을 수정하여 csv 파일 및 json 저장경로를 지정할 수 있습니다.
```python
root = ['*.csv',] # 앙상블을 진행할 csv 파일을 지정합니다.
submission_dir = '../../submission/' # csv 파일이 저장된 경로 및 앙상블 후 저장할 경로를 지정합니다.
annotation = '../../dataset/test.json' # 앙상블에 사용하기 위해 file의 image 정보가 포함된 json 파일 경로를 지정합니다.
```

아래 변수 값을 수정하여 앙상블 기법 및 수치를 지정할 수 있습니다.
```python
ensemble_type = '' #[nms, wbf, nmw, soft-nms] 중 사용할 앙상블 기법을 선택합니다. 
iou_thr = 0.5 #iou threshold 값을 설정합니다.

# WBF 기법 설정 값
wbf_conf_type='avg' # ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'] # WBF 기법 수행 시 신뢰도 계산 방법을 설정 값입니다.
wbf_allows_overflow = False # {True: 가중치 합 > 1, False: 가중치 합 1로 고정} # 가중치 합을 1을 초과하거나 1로 고정 하는 설정 값입니다.
wbf_skip_box_thr = 0.0 # 값에 해당하는 정확도가 넘지 않으면 제외하는 설정 값입니다.

# Soft-NMS 기법 설정 값
method = 2 # 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS 기본값: 2  # Soft-NMS의 방식을 선택하는 설정 값입니다.
sn_sigma = 0.5 # Gaussian soft-NMS 방식 사용 시 분산을 설정하는 값입니다. 
sn_thresh = 0.001 # 값에 해당하는 신뢰도 미만의 Box를 제거하는 설정 값입니다.


weights = [1] * len(submission_df) # 각 모델의 동일한 가중치 1을 고정하는 설정 값입니다. None으로 설정 시 각 모델에 적용된 가중치로 진행됩니다. 

```

해당 코드들은 Weighted-Boxes-Fusion GitHub 내 ensemble_boxes 라이브러리가 포함되어 있습니다.
- [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)  

## 🧑‍🤝‍🧑 Contributors
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/Yeon-ksy"><img src="https://avatars.githubusercontent.com/u/124290227?v=4" width="100px;" alt=""/><br /><sub><b>김세연</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/jihyun-0611"><img src="https://avatars.githubusercontent.com/u/78160653?v=4" width="100px;" alt=""/><br /><sub><b>안지현</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/dhfpswlqkd"><img src="https://avatars.githubusercontent.com/u/123869205?v=4" width="100px;" alt=""/><br /><sub><b>김상유</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/K-ple"><img src="https://avatars.githubusercontent.com/u/140207345?v=4" width="100px;" alt=""/><br /><sub><b>김태욱</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/myooooon"><img src="https://avatars.githubusercontent.com/u/168439685?v=4" width="100px;" alt=""/><br /><sub><b>김윤서</b></sub><br />
    </td>
  </tr>
</table>
</div>

## ⚡️ Detail   
프로젝트에 대한 자세한 내용은 [Wrap-Up Report](https://github.com/boostcampaitech7/level2-objectdetection-cv-23/blob/main/docs/CV_23_WrapUp_Report_detection.pdf) 에서 확인할 수 있습니다.
