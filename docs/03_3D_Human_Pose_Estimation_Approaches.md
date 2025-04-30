# 3D Human Pose Estimation 접근법

## 목차
1. [3D Human Pose Estimation의 기초](#1-3d-human-pose-estimation의-기초)
2. [3D Human Pose Estimation의 분류](#2-3d-human-pose-estimation의-분류)
3. [Model-based 접근법](#3-model-based-접근법)
4. [Model-free 접근법](#4-model-free-접근법)

---

## 1. 3D Human Pose Estimation의 기초

### 2D vs. 3D: 시점 변경 가능성
- **2D 표현:** 특정 시점(view)의 이미지에 국한되어 다른 시점에서 대상을 볼 수 없음
- **3D 표현:** 대상의 3차원 정보를 담고 있어 임의의 시점에서 자유롭게 대상을 표현 가능

### 3D Human Pose 표현 방식
- **3D 관절 좌표 (3D Joint Coordinates):** 
  - 각 관절의 3D 공간 상의 (x, y, z) 위치
  - 이 정보만으로는 사람의 표면(surface, mesh) 표현 불가능

- **3D 관절 회전 (3D Joint Rotation):** 
  - 각 관절의 회전 정보
  - 3D Human Model과 함께 사용하면 사람의 3D 표면(mesh) 표현 가능

- **3D Human Mesh:**
  - 많은 수의 작은 삼각형 집합으로 3D 물체의 표면을 표현
  - 꼭지점(vertex)과 면(face)으로 구성
  - Human Pose Estimation에서는 **꼭지점들의 3D 좌표**를 구하는 것이 목표

- **3D 관절 회전 + 3D 길이/체형:** 
  - 동일한 관절 회전이라도 사람의 길이(뼈 길이)나 체형에 따라 최종 3D 형태가 달라짐
  - 완전한 3D pose 정의에는 관절 회전과 길이/체형 정보 모두 필요

### 3D Human Model
- **정의:** 3D 관절 회전(θ)과 다른 파라미터들(β, 예: 길이/체형)을 입력받아 사람의 3D mesh를 출력하는 함수

#### θ (3D 관절 회전, J x 3)
- 해당 관절의 **부모 관절에 상대적인** 3D 회전
- 회전은 자식 관절들의 위치에 영향을 줌
- 'leaf node' 관절(손가락 끝 등)은 일반적으로 회전이 정의되지 않음
- 'root node'(골반 등) 관절의 회전은 전신(global) 회전에 해당

#### β (3D 길이 및 체형)
- T-pose(모든 관절 회전이 0인 기본 자세)를 취한 사람의 길이와 체형 파라미터
- 다양한 사람의 3D 스캔 데이터를 **주성분 분석(PCA)** 하여 체형 공간 모델링
- PCA 계수(coefficient)를 β로 사용

#### Linear Blend Skinning (LBS)
- **정의:** T-posed mesh에 관절 회전을 적용하여 최종 mesh를 만드는 알고리즘
- **Skinning Weight (W, V x J):** 각 mesh 꼭지점이 어떤 관절의 회전에 얼마나 영향받는지 정의
- **작동 방식:** 모든 관절의 변형 행렬을 각 꼭지점의 skinning weight에 따라 선형으로 합성
- **전체 Pipeline:** β (PCA) → T-posed mesh → θ + LBS → 최종 3D mesh

#### Pose-dependent Correctives
- LBS만으로는 팔꿈치 접힘 등 특정 자세의 미세하고 복잡한 표면 변화 표현 어려움
- 특정 자세에 따라 추가적으로 적용되는 변형 정보(blend shapes) 사용

### 대표적인 3D Human Body Models

#### 1. SMPL (Skinned Multi-Person Linear Model)
- 가장 널리 사용되는 전신(body) 모델
- **구성 요소:** T-pose, Shape blend shapes (Bs), Pose blend shapes (Bp), Joints regressor, LBS
- MultiPose (pose)와 MultiShape (shape) 같은 3D 스캔 데이터셋으로 제작

#### 2. MANO (hand model)
- 손 모델
- SMPL과 유사한 구성 요소(T-pose, Shape, Pose, LBS) 보유
- 손가락 움직임의 복잡성으로 **pose space에도 PCA 적용**하여 유효한 회전 공간 모델링

#### 3. SMPL-X (whole-body model)
- SMPL (body), MANO (hand), FLAME (face) 모델을 결합한 전신 모델
- θ (body, hand), β (shape), ψ (얼굴 표정) 파라미터를 입력으로 사용

#### SMPLify-X (Fitting SMPL-X to 2D Pose)
- 2D pose estimator로 얻은 2D pose에 대해 **Energy Minimization** 방식으로 SMPL-X 파라미터를 찾는 프레임워크
- Energy 함수는 2D pose 데이터 텀과 정규화 항(regularizer)으로 구성
- **단점:** 느린 속도, 잘못된 2D pose나 depth ambiguity에 취약
- **Vposer:** 해부학적으로 불가능한 3D 관절 회전 방지를 위해 Pose Space에 VAE 적용

---

## 2. 3D Human Pose Estimation의 분류

### Model-Based 접근법 vs. Model-Free 접근법

#### Model-based
- **정의:** Neural Network가 입력 이미지로부터 **3D Human Model의 파라미터(θ, β 등)**를 추정
- **출력:** 추정된 파라미터를 3D Human Model에 통과시켜 최종 3D mesh/joints 계산
- **특징:** 
  - 3D 관절 회전(θ) 추정으로 부모 관절 에러가 자식 관절로 누적되는 **Error Accumulation** 발생 가능
  - 좌표 정확도는 다소 낮을 수 있으나, 추정된 3D 관절 회전은 다양한 응용에 유용

#### Model-free
- **정의:** Neural Network가 입력 이미지로부터 **Human Mesh 꼭지점들의 3D 좌표**를 직접 추정
- **출력:** 3D Human Model의 topology를 따르는 3D mesh (V x 3 형태의 좌표)
- **특징:**
  - Error Accumulation 현상이 없어 좌표 정확도가 높을 수 있음
  - 계산량(computational overhead)이 많고 추정된 mesh가 부드럽지 않을 수 있음
  - 3D 관절 회전 값을 직접 얻을 수 없어 응용에 제약

### Model-Based 접근법의 학습
- **학습 데이터:** In-the-wild dataset (GT 2D pose) + MoCap dataset (GT 3D pose/parameters)

#### In-the-wild 데이터 학습
- **문제:** 2D GT만 있고 3D GT는 없음
- **해결:** 추정된 3D mesh/joints를 가상 카메라로 입력 이미지 2D 공간에 **Project**
- **과정:**
  1. 추정된 3D mesh에서 3D 관절 좌표 추출 (Model의 Joint Regressor 사용)
  2. 가상 카메라 파라미터 및 3D translation vector 추정
  3. 카메라 파라미터로 3D 관절 좌표를 2D 이미지 공간에 project
  4. Project된 2D 관절 좌표와 GT 2D pose 사이의 **L1 Loss** 계산

#### MoCap 데이터 학습
- **문제:** 3D GT pose/parameters 존재
- **해결:** 추정된 3D mesh에서 3D 관절 좌표를 추출하고 GT 3D pose와 비교
- **과정:**
  1. 추정된 3D mesh에서 3D 관절 좌표 추출
  2. 추출된 3D 관절 좌표와 GT 3D pose 사이의 **L1 Loss** 계산

### Model-Free 접근법의 학습
- **학습 데이터:** In-the-wild 혹은 MoCap dataset의 GT 2D/3D 관절 좌표
- **문제:** Model-free는 3D mesh vertex GT가 필요하나, 일반 데이터셋은 관절 GT만 제공
- **해결:** 학습을 위한 GT로 **3D Pseudo-GT Mesh** 사용
- **과정:**
  1. Pre-processing 단계에서 GT 2D/3D pose로 **3D Pseudo-GT Mesh** 생성
  2. 네트워크 출력 3D mesh와 3D Pseudo-GT mesh 사이의 **L1 Loss** 계산

---

## 3. Model-based 접근법

### 1. HMR (Human Mesh Recovery, CVPR 2018)
- 가장 초기 Model-based 방법 중 하나

#### 파이프라인
- Image → Encoder (ResNet 등) → Regression (FC layers) → SMPL parameters (θ, β) + Camera parameters
- SMPL Model → 3D Mesh/Joints → Orthogonal Projection → Projected 2D Joints

#### 학습
- MoCap 데이터의 3D GT와 In-the-wild 데이터의 2D GT 활용
- **Loss Functions:**
  - `Lreproj`: Projected 2D joints와 GT 2D pose 간의 L1 Loss (2D Supervision)
  - `L3D`: Predicted 3D joints/parameters와 GT 3D joints/parameters 간의 L2 Loss (3D Supervision)
  - `Ladv`: Adversarial Loss - Depth Ambiguity 문제 해결을 위한 핵심 기법
    - Discriminator가 Predicted SMPL Parameters의 진위 여부 판별
    - Encoder는 Discriminator를 속이도록 학습

### 2. Pose2Pose (2020)

#### 기존 방식의 한계
- Global Average Pooling (GAP)은 이미지의 spatial domain 정보를 손실
- 관절별 semantic 정보 활용이 어려움

#### 제안 방법
- 네트워크가 먼저 **3D 관절 좌표(x,y: pixel, z: root-relative depth)**를 추정
- 추정된 관절 (x,y) 위치로 Bilinear Interpolation을 통해 **Joint-specific Image Feature** 추출 (PPP)
- Joint Feature와 3D 관절 좌표를 활용하여 **3D 관절 회전 및 SMPL 파라미터** 추정
  - Joint Feature: contextual information 제공
  - 3D 관절 좌표: explicit geometry information 제공

### 3. HybrIK (Hybrid Inverse Kinematics, CVPR 2021)

#### Motivation
- Neural Network가 3D 관절 회전 전체를 직접 추정하는 것은 어려움
- 예측하기 쉬운 파라미터만 예측하고, 분석적 모듈로 전체 3D 회전 복원

#### FK vs. IK
- **FK (Forward Kinematics):** 3D 관절 회전 → 3D 관절 좌표 계산
- **IK (Inverse Kinematics):** 목표 3D 관절 좌표 → 3D 관절 회전 계산 (ill-posed problem)

#### 제안 방법
- CNN이 3D 관절 좌표(P)와 Twist 회전(Φ, 1D 각도)을 예측
- **HybrIK** 모듈로 이 정보에서 전체 3D 회전(θ) 복원
- Beta(shape)는 별도 FC layer로 추정

#### HybrIK 모듈 상세
- **Twist-and-Swing Decomposition:** 
  - 모든 3D 회전은 Swing(평면 내 회전)과 Twist(축 회전)로 분해 가능
  - Swing은 3D 관절 좌표 변화에서 유도 가능
  - Twist는 네트워크가 직접 예측

- **Naive HybrIK:** 
  - 분해된 회전을 SMPL 필요 3D 상대 회전으로 변환
  - T-pose와 예측 자세의 뼈 길이 동일 가정

- **Adaptive HybrIK:** 
  - Naive HybrIK 가정 문제 해결
  - Error accumulation 방지를 위해 예측 3D 관절 좌표 타겟 적응적 업데이트

---

## 4. Model-free 접근법

### 1. I2L-MeshNet (Image-to-Lixel Prediction Network, ECCV 2020)

#### Motivation
- Voxel 기반 3D heatmap은 높은 성능을 보이지만 메모리 사용량이 큼 (O(VD³))
- Mesh의 많은 vertex에 대해 voxel heatmap 추정 시 메모리 문제 심각

#### 제안 방법
- **Lixel (Line + Pixel)** 기반 1D heatmap 개념 제안
- 메모리 복잡도를 O(VD)로 감소

#### 파이프라인
- Image → Regressor → Mesh vertex마다 **세 개의 Lixel 기반 1D heatmap(x, y, z축)** 추정
- 각 heatmap에 **Soft-argmax(미분 가능)** 적용하여 vertex의 3D 좌표 추출

### 2. Pose2Mesh (ECCV 2020)

#### Motivation
- Image 기반 방법들은 MoCap 학습 데이터와 In-the-wild 테스트 데이터 간 **Appearance Domain Gap** 문제 발생
- **2D Pose**를 입력으로 사용하여 이 문제 해결

#### 2D Pose 입력 사용의 장점
- 2D Pose는 이미지의 Geometry 정보를 잘 표현
- MoCap 3D 데이터와 In-the-wild 2D 데이터 간 **Geometry Domain Gap**은 Appearance Gap보다 작음
- 2D Pose는 In-the-wild 이미지에서도 잘 추정되거나 쉽게 annotation 가능

#### 파이프라인
- 입력: 2D Pose → PoseNet → Coarse 3D Pose → MeshNet (GraphCNN) → **Coarse-to-fine Mesh Estimation**
- MeshNet은 Graph Convolutional Network로 Mesh 구조 활용

### 3. METRO (MEsh TRansfOrmer, CVPR 2021)

#### Motivation
- Transformer는 입력 토큰들 간 관계(correlation)를 효과적으로 모델링
- 관절과 Mesh vertex들을 "토큰"으로 간주하여 복잡한 상호 관계 모델링 가능

#### 파이프라인
- Image → CNN → Image Feature → Input Queries (Image feature + T-pose 3D joints/vertices 위치 인코딩)
- Multi-Layer Transformer Encoder → 3D coordinates for joints and vertices

#### 핵심 특징
- Transformer의 Self-Attention 메커니즘으로 **모든 관절/vertex 쌍 사이의 상호 관계** 학습
- **Masked Vertex Modeling (MVM):** 
  - 입력 Query 중 일부를 랜덤하게 Masking하는 Data Augmentation 기법
  - Transformer가 Mask 부분 예측을 위해 다른 Query 간 관계를 더 잘 활용하도록 유도
  - Robustness 향상에 기여
