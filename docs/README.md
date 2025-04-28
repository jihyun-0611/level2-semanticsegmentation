# 📈 Semantic Segmentation 프로젝트 (개인 기여 정리)

## 프로젝트 개요

- 의료 영상(손 뼈 이미지)에서 여러 개 클래스가 겹치는 픽셀을 정밀하게 분할하는 세그멘테이션 문제

## 사용 데이터

- 고해상도(2048x2048) 손 뼈 X-ray 이미지
- 총 29개 손 뼈 구조를 분류해야 하는 멀티 클래스 세그멘테이션

## 주요 기여

### 1. 모델 실험

- **EffiSegNet 기반 실험**
  - [model](https://github.com/jihyun-0611/level2-semanticsegmentation/blob/main/models/EffiSegNet.py) | [experiments](https://github.com/jihyun-0611/level2-semanticsegmentation/blob/main/experiments/completed/31.6_EffiSegNet_b7_BCE.yaml)
  - EfficientNet을 인코더로 사용하고 SegNet 구조를 디코더로 적용한 **EffiSegNet**을 실험.
  - 구조가 단순하여 학습 속도가 빠르기 때문에 다양한 실험이 가능했고, 기존 Medical Segmentation에서 SOTA 2위 수준의 성능을 가지고 있었음.
  - 결과
    - Dice Score : 0.9628
    - 전체적으로 빠르고 안정적인 학습이 가능했고, 주요 클래스에 대해 준수한 Dice Score를 기록(최대 0.96921).
    - 다만, Pisiform 등 작은 객체나 클래스 중첩이 있는 영역에서는 Dice Score 하락 및 예측 편차가 크게 나타남.
    - DiceFocalLoss 사용 시 작은 객체 성능이 불안정해졌고, BCEWithLogitsLoss를 사용할 때 비교적 더 안정적인 결과.
    - 디코더가 nearest neighbor 업샘플링 후 단순 덧셈으로 특징을 결합하는 방식으로 설계되어 인코더 모델 크기를 키워도 작은 객체나 세부 구조 복원에는 한계가 있음.
- **SwinUNETR 기반 실험**
  - [model](https://github.com/jihyun-0611/level2-semanticsegmentation/blob/main/models/SwinUNETR.py) | [experiments](https://github.com/jihyun-0611/level2-semanticsegmentation/blob/main/experiments/completed/36.4_Finetune_SwinUNETR_slidingwindow.yaml)
  - EffiSegNet 기반 실험에서 작은 객체, 다중 클래스 구간에서 정보 손실이 발생하는 문제를 확인.
  - Swin Transformer 기반 인코더와 점진적으로 해상도를 복원하는 디코더를 가진 **SwinUNETR** 모델로 세부 특징 보존 및 전역적 관계 파악을 강화하고자 함.
  - Sliding Window를 활용한 고해상도 이미지 학습(2048x2048)으로 다중 클래스가 겹치는 픽셀 처리를 개선하는 것을 목표로 함.
  - 결과
    - Dice Score : 0.9716
    - 작은 뼈 구조(Pisiform 등)에서 Dice Score 향상 확인
    - 다중 클래스 중첩 구간에서 예측 안정성 개선
    - 경계 오류율(Boundary Error Rate) 감소
    - Confusion Matrix 분석 결과, 주요 클래스 간 오분류 감소

### 2. 앙상블 실험

#### 실험 목적

- 전체 손등뼈 구조에 대해 높은 평균 Dice Score를 가진 모델과,
- 다중 클래스를 가진 작은 뼈 구조(Pisiform 등)에 강점을 보이는 모델을 별도로 선별하여 앙상블을 진행
- 각각의 강점을 살려 전체적인 분할 성능과 작은 구조 분할 성능을 모두 개선하는 것을 목표로 함.

#### 앙상블 구성

- **일반적으로 평균 DICE 점수가 우수한 모델** 3개
- **작은 구조 성능이 우수한 모델** 2개
```
- 36_4_SwinUNETR_finetuning_swa
- 37_SWA_Sliding_Tver
- 37_SWA_Sliding_window
- 42_0_segformer_newdata
- 37_U++_HR_FocalTversky
```
- 총 5개 모델을 앙상블하여 최종 예측 결과 생성


- Public 기준 최종 Dice Score: **0.9755** 달성
- 기존 단일 모델 대비 전체 평균 성능과 클래스 중첩 영역 모두에서 성능 향상



### 3. 실험 관리 체계 구축

- **Wandb 자동화 연동**
  - 학습 로그를 wandb로 자동 기록
  - 실험별 학습 곡선, 성능 지표를 체계적으로 관리 가능하게 개선
- **노션 실험 기록 템플릿 설계**
  - 실험 가설, 변경된 설정, 결과 요약을 구조화된 템플릿으로 기록하도록 가이드
  - 팀원들의 실험 기록 품질 및 일관성 향상
- **오류 분석 도구 작성**
  - 실험 결과를 정밀 분석하기 위한 시각화 도구 구현
  - [https://github.com/jihyun-0611/level2-semanticsegmentation/tree/main/error_analysis](https://github.com/jihyun-0611/level2-semanticsegmentation/tree/main/error_analysis)
    - Confusion Matrix
    - Dice Score Distribution
    - Boundary Error Rates

---


> **본 문서는 개인 기여 내용을 중심으로 작성되었습니다.**  

