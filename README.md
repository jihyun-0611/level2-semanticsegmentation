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
데이터셋은 한 사람의 양 손을 촬영한 X-Ray 이미지이며, 하나의 ID에 한 사람에 대한 오른손 및 왼손의 이미지가 들어있습니다. 각 이미지는 의학적으로 분류된 손 뼈 29가지의 클래스를 가지며, 2048x2048 크기의 train 이미지 800장, test 이미지 288장으로 구성됩니다.

또한, 각 ID에 해당하는 사람의 나이, 성별, 체중, 신장에 대한 Mata_data가 xlsx로 주어집니다. 

### Train json

Train json 파일은 각 이미지에 대한 annotations이 포함되며, id, type, attributes, points, label로 구성되어 있습니다.
- Images
  ```json
    {"annotations": [{
        "id": "id", 
        "type": "poly_seg", 
        "attributes": {}, 
        "points": [[[10, 20], [30, 40], ... ]], 
        "label": "finger-1"} 
        ...]}
  ```
  - ⚠️ 데이터 보안상의 문제로 ID와 Points는 임의로 작성되었습니다.
<br />

## ⚙️ Requirements

### env.
이 프로젝트는 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 훈련 및 테스트되었습니다.

### Installation
이 프로젝트에서는 다양한 라이브러리가 필요합니다. 아래 단계를 따라 필요한 라이브러리를 설치하세요.

#### 1. PyTorch 설치

PyTorch 2.1.0을 설치합니다. 설치 방법은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)를 참고하세요.

#### 2. 프로젝트 의존성 설치

다음 명령어를 실행하여 프로젝트를 클론하고 필요한 의존성을 설치합니다:

```bash
git clone https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3.git
cd level2-cv-semanticsegmentation-cv-04-lv3
pip install -r requirements.txt
```

<br />

## 🎉 Project
### 1. Structure
```bash
Project
├── base_config.yaml
├── config
│   ├── config.py
├── data
│   ├── augmentation.py
│   └── dataset.py
├── eda_and_visualization
│   ├── confusion_matrix.py
│   ├── EDA.ipynb
│   └── visualize_csv.ipynb
├── ensemble.py
├── ensembles
│   ├── hard_voting.py
│   ├── __init__.py
│   └── soft_voting.py
├── error_analysis
│   ├── analysis_from_wandb.ipynb
│   ├── analysis_smp_encoder_decoder.ipynb
│   ├── confusion_matrix.py
│   ├── error_analysis.py
│   └── evaluation.py
├── experiments
│   ├── completed
│   │   ├── 10_completed_resnext101_32x8d.yaml
│   │   ├── .....
│   ├── ensemble.yaml
│   ├── README.md
├── inference.py
├── inference.sh
├── mmsegmentation
├── models
│   ├── base_model.py
│   ├── DeepLabV3Plus.py
│   ├── ....
├── multi_inference.sh
├── multi_train.sh
├── README.md
├── requirements.txt
├── train.py
├── train.sh
└── utils
    ├── convert_format.py
    ├── download_artifacts.py
    ├── loss.py
    ├── metrics.py
    ├── optimizer.py
    ├── scheduler.py
    └── utils.py
```
### 2. Train & Inference
이번 대회에서는 실험 구현, 공유, 그리고 재현성을 높이기 위해 YAML 파일을 활용하여 실험 파라미터 설정 및 관리하는 방식을 도입했습니다. 우리가 사용한 기본 설정 파일은 [base_config.yaml](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/blob/main/base_config.yaml)에서 확인할 수 있습니다. 실제 실험에 사용된 YAML 파일들은 [여기](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/tree/main/experiments/completed)에서 확인 가능합니다. 또한, 우리의 YAML 파일 작성 규칙은 [이곳](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/tree/main/experiments)에서 자세히 확인할 수 있습니다.

또한, 다양한 모델과 커스텀 모델을 사용했습니다. 모델 작성 규칙은 [다음](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/tree/main/models)에서 확인할 수 있습니다.


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
        <td align="center"><a href="https://github.com/cherry-space"><img src="https://avatars.githubusercontent.com/u/177336350?v=4" width="100px;" alt=""/><br /><sub><b>김채리</b></sub><br />
  </tr>
</table>
</div>

## ⚡️ Detail   
우리는 해당 대회를 위해 다양한 모델과 방법론을 적용하였으며, 이에 대한 자세한 내용은 [Wrap-Up Report](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/blob/main/docs/SemanticSeg_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(04%EC%A1%B0).pdf)에서 확인하실 수 있습니다.

