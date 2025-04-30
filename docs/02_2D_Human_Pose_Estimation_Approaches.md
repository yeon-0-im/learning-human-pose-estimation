# 2D Human Pose Estimation 접근법

## 목차
1. [2D Human Pose Estimation의 분류](#1-2d-human-pose-estimation의-분류)
2. [Top-down 접근법](#2-top-down-접근법)
3. [Bottom-up 접근법](#3-bottom-up-접근법)

---

## 1. 2D Human Pose Estimation의 분류

### Multi-Person Pose Estimation 목표
- 입력 이미지로부터 여러 사람의 관절을 2D 공간에 정확하게 위치화

### Top-down vs. Bottom-up 접근법
- **Top-down 접근법:**
  - **단계:** Human detection → Single person pose estimation
  - 사람 수만큼 pose estimation 네트워크 실행 필요
  
- **Bottom-up 접근법:**
  - **단계:** Joint detection → Grouping (관절들을 같은 사람끼리 묶기)
  - 사람 수와 관계없이 pose estimation 네트워크는 한 번만 실행

### 두 접근법 비교
- **Top-down:**
  - **정확성:** 일반적으로 더 높은 정확성 (예: MSCOCO AP 78 vs 71)
    - 뛰어난 Human detection 네트워크 활용
    - 고해상도 사람 영역을 입력으로 사용
  - **효율성:** 사람 수에 비례하여 연산량 증가 (많은 사람이 있을 때 비효율적)

- **Bottom-up:**
  - **정확성:** Top-down보다 다소 낮음
    - 전체 이미지에서 관절을 찾아 저해상도 문제 발생
    - Grouping 과정에서 오류 가능성
  - **효율성:** 사람 수에 크게 영향 받지 않음 (많은 사람이 있을 때 효율적)

---

## 2. Top-down 접근법

### Top-down Pipeline 상세
- **기본 Pipeline:** 
  - Human detector로 사람 bounding box 검출
  - 각 box 영역을 crop하여 Single person pose estimator의 입력으로 사용

- **Mask R-CNN의 개선점:**
  - 객체 탐지, 분할, keypoint 탐지를 통합한 모델
  - 입력 이미지에서 직접 crop하는 대신 Feature map에서 **RoIAlign** 연산 사용
  - **RoIAlign:** 
    - Bilinear interpolation을 통해 특징 추출 (미분 가능한 방식)
    - RoIPool의 discretization 문제 해결
    - Detection과 pose estimation 네트워크를 **end-to-end (e2e)** 학습 가능
  - JxHxW 형태의 2D heatmap 추정

### 대표적인 Single Person Pose Estimator

#### 1. SimpleBaseline (ECCV 2018)
- **구조:** ResNet backbone + 3개의 deconvolutional layer
- **특징:** 비교적 단순한 네트워크 구조
- **성과:** 단순함에도 불구하고 당시 SOTA 수준 달성
- **추천:** YOLOv5 Human detector와 함께 사용 시 구현 용이

#### 2. HRNet (High-Resolution Net) (CVPR 2019)
- **배경 문제:** 
  - 기존 네트워크는 입력 이미지를 크게 downsampling (예: 32배)
  - 작은 신체 부위 정보 손실이나 discretization 문제 발생
  
- **HRNet 특징:** 
  - 네트워크 전체에 걸쳐 **고해상도 feature map 보존**
  
- **구조:** 
  - 여러 해상도의 feature map을 병렬적으로 유지
  - 다른 해상도의 feature map 간 정보 교환 (Multi-scale feature fusing)
  
- **성과:** 
  - 고해상도 feature와 Multi-scale fusing으로 정확도 크게 향상
  - 다른 SOTA 모델들보다 높은 정확도 달성 (예: AP 74.4 vs 70.4)
  - 비슷한 연산량(GFLOPs)에서도 우수한 성능
  - 가장 널리 사용되는 backbone 중 하나로 자리매김

---

## 3. Bottom-up 접근법

### Bottom-up Pipeline 상세
- **기본 Pipeline:** 
  - Joint detector로 이미지 전체에서 모든 관절 후보 탐지
  - Grouping 과정을 통해 각 관절을 특정 사람에게 할당

#### 1. Associative Embedding (NeurIPS 2017)
- **Joint Detection:** 
  - 2D Gaussian heatmap을 추정하여 모든 관절 위치 탐지
  
- **Grouping (Associative Embedding):**
  - **개념:** 각 관절이 어떤 사람에 속하는지 결정하는 방법
  - **Associative Embedding Map:** 
    - Detection heatmap 외에 "tag value" embedding map 추가 출력
    - Tag value는 각 픽셀이 어떤 사람에 속하는지 암묵적으로 표현
  
  - **학습 목표:** 
    - 같은 사람에 속하는 관절들은 **비슷한 tag value** 갖도록 학습
    - 다른 사람에 속하는 관절들은 **다른 tag value** 갖도록 학습
    - Loss 함수: 그룹 내 분산 최소화, 그룹 간 거리 최대화
  
  - **Grouping 과정:** 
    - 탐지된 관절들의 tag value 차이가 임계값보다 작으면 동일 인물로 판정
  
  - **성과:** 
    - Bottom-up 방식에서도 여러 사람의 pose 성공적 추정
    - 당시 기존 Bottom-up 방법들보다 우수한 성능 달성

#### 2. HigherHRNet (CVPR 2020)
- **Motivation:** 
  - Top-down의 HRNet처럼 고해상도 feature map의 중요성 인식
  - 다양한 스케일의 사람(크기가 큰/작은 사람)이 혼재된 환경 고려
  - **Scale-aware** 추정의 중요성 강조
  
- **HigherHRNet 특징:** 
  - HRNet을 backbone으로 사용
  - **Multi-resolution supervision:** 
    - 여러 해상도(예: 1/4, 1/2 스케일)의 heatmap 동시 추정
  - **Multi-scale fusion:** 
    - 여러 해상도의 heatmap 조합으로 정확도 향상 (Heatmap Aggregation)
  
- **Grouping:** 
  - Associative Embedding 방법 사용하여 관절 그룹핑
  
- **성과:** 
  - Multi-resolution supervision이 스케일 강건성에 필수적임을 입증
  - 기존 Bottom-up 방법들 중 최고 성능 달성
  - 하지만 Top-down SOTA 모델들(HRNet-W48 등)과는 여전히 성능 차이 존재
