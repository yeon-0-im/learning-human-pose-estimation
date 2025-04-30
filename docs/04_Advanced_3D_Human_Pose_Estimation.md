# 고급 3D Human Pose Estimation 주제

## 목차
1. [여러 사람의 3D Pose Estimation](#1-여러-사람의-3d-pose-estimation)
2. [상호작용하는 두 손의 3D Pose Estimation](#2-상호작용하는-두-손의-3d-pose-estimation)
3. [비디오로부터의 3D Pose Estimation](#3-비디오로부터의-3d-pose-estimation)
4. [3D Whole-Body Pose Estimation](#4-3d-whole-body-pose-estimation)
5. [3D Human Pose Estimation의 성과와 후속 연구](#5-3d-human-pose-estimation의-성과와-후속-연구)

---

## 1. 여러 사람의 3D Pose Estimation

### 단일 사람 vs. 여러 사람의 3D Pose Estimation
- **단일 사람 3D Pose Estimation:** 
  - 한 사람에 대한 **Root Joint-relative 3D 관절 좌표** 추정
  - Global translation 정보 제거로 네트워크가 자세 자체에 집중

- **여러 사람 3D Pose Estimation:** 
  - 여러 사람 간 상대적 위치 관계 파악을 위해 **Absolute 3D Pose** 필요
  - 2D Multi-Person과 달리 **3D Human Root Localization** 과정 추가 필요

### 기존 3D Multi-Person Pose Estimation 방식
- **3D-to-2D Fitting:**
  - 단일 사람의 root-relative 3D pose 추정 후 절대적 3D 위치 획득
  - 추정된 3D pose를 project 시킨 2D pose와 기존 추정된 2D pose 간 거리 최소화
  - **한계점:** 추정된 2D/3D pose 자체의 오류에 민감하며 불안정

### 3DMPPE Framework
- **특징:** 3D-to-2D fitting 없이 **Fully Learning-based** 접근법

#### 구성 요소
- **DetectNet:** Human Detection Network (Mask R-CNN 등)
- **PoseNet:** Root-relative 3D Single Person Pose Estimation Network
- **RootNet:** Root Joint Localization Network
  - **핵심 문제: Scale Ambiguity** - 단일 이미지에서 절대적 깊이 추정 어려움
  - **해결책: Initial Depth Value k + Correction Factor γ**
    - Camera Pinhole Model로 Initial Depth Value `k` 계산
    - Image Feature 기반 Correction Factor `γ` 추정
    - Absolute Depth ZR = k × √γ
    - `γ`는 focal length와 무관해 다양한 이미지에 적용 가능

#### 성능 및 특징
- RootNet과 PoseNet 성능이 Absolute 3DMPPE 정확도에 큰 영향
- 기존 Fitting 방식보다 root depth 추정 에러 크게 감소
- Focal length 정보가 없는 이미지에서도 효과적

---

## 2. 상호작용하는 두 손의 3D Pose Estimation

### 단일 손 vs. 상호작용하는 두 손 3D Pose
- **단일 손 3D Pose:** 한 손 이미지에서 21개 관절의 3D 위치 추정
- **상호작용하는 두 손 3D Pose:** 접촉하거나 얽혀 있는 두 손 이미지에서 각 손의 3D Pose 추정

### 기존 3D Hand Pose Datasets의 한계
- 대부분 Single Hand만 포함
- 두 손 포함 데이터셋도 상호작용 없거나 규모 작음
- 합성 데이터이거나 비공개인 경우 많음

### InterHand2.6M (ECCV 2020)
- **의의:** 최초의 대규모(2.6M 프레임) Real RGB 이미지 + 정확한 GT 3D Pose 데이터셋
- **데이터셋 구축:**
  - **Capture Studio:** 약 100대 동기화된 카메라, 고해상도/고프레임 촬영
  - **Subjects:** 26명(남/녀 포함)
  - **Hand Sequences:** Peak Pose와 Range of Motion 시퀀스
  - **Semi-Automatic Annotation:**
    - Manual Annotation: Multi-view consistency 확보
    - Automatic Annotation: 학습된 Detector와 RANSAC 기반 Triangulation
- **Splits:** Human-annotated(H)와 Machine-annotated(M)로 구분

### InterNet: 3D Interacting Hand Pose Estimation Network
- InterHand2.6M 데이터셋 위한 Baseline 모델
- **출력:** Handedness, 2.5D Hand Pose, Right hand-relative Left hand Depth
- **학습:** BCE Loss(Handedness), L2 Loss(2.5D Pose), L1 Loss(상대 깊이)
- **테스트:** 예측 결과 조합 + Hand Root Depth로 Absolute 3D Interacting Hand Pose 계산
- **성능:**
  - Interacting Hand 데이터 사용 시 성능 크게 향상
  - Single Hand Pose Estimation에서 기존 SOTA 모델 대비 우수한 성능
  - Depth map 기반보다 에러는 높지만 RGB 이미지에서는 획기적 성과

---

## 3. 비디오로부터의 3D Pose Estimation

### 단일 이미지 vs. 비디오 기반 3D Pose Estimation
- **단일 이미지:** 특정 순간의 이미지에서 3D Pose 추정
- **비디오 입력의 필요성:** 실제 대부분 애플리케이션은 비디오 입력 사용
- **단일 이미지 방법의 비디오 적용 문제:**
  - Motion Blur 심한 프레임에서 정확도 저하
  - 시간적 일관성(Temporal Consistency) 부족

### Temporal Consistency 문제 및 해결
- **Temporally Inconsistent Results:** 기존 비디오 기반 방법도 프레임 간 포즈 변화 부자연스러움
- **원인:** Static Feature에 과도하게 의존, 각 프레임 Pose 에러 방향의 비일관성

#### TCMR (Temporally Consistent Mesh Recovery, CVPR 2021)
- **목표:** Static Feature 의존도 낮춰 Temporal Consistency 개선
- **방법:**
  - Static과 Temporal Feature 사이 Residual Connection 제거
  - **PoseForecast:** 과거/미래 프레임으로만 현재 프레임 pose 예측
  - Static Feature 의존도 감소로 Acceleration Error 크게 개선

### GPU 메모리 문제 해결
- **입력을 2D Pose로 대체:** 이미지 대신 2D Pose 시퀀스 사용하여 메모리 절약
- **Image Feature Vector 미리 추출:** Feature를 디스크에 저장하고 재사용

### 성능 및 특징
- 비디오 기반 방법이 단일 이미지 기반보다 낮은 에러 기록
- TCMR은 다른 비디오 기반 방법 대비 Acceleration Error 크게 감소
- 정성적으로도 시간적으로 훨씬 부드럽고 일관된 움직임 생성

---

## 4. 3D Whole-Body Pose Estimation

### Body-Only vs. Whole-Body 3D Pose Estimation
- **Body-Only/Hand-Only:** 기존에는 Body, Hand, Face 별도 추정 후 Integration
- **3D Whole-Body:** Body, Hand, Face 포함한 전신을 한 번에 3D 공간에 복원

### 3D Whole-Body Pose Estimation의 어려움
- **Hand 복원 문제:**
  - Complicated hand articulations: 복잡한 손가락 관절 움직임
  - Small hand areas: 전신 이미지에서 손 영역이 매우 작음
  - Many hand joints: 손 관절 수(21개)가 Body 관절(약 20개)보다 많음

### Integration 문제
- Body, Hand, Face 모델 단순 연결 시 팔목(Wrist) 등에서 부자연스러운 결과
- Accurate 3D Wrist가 매우 중요

### Hand4Whole (2020)
- **목표:** 정확한 3D Hand Pose(특히 Wrist, Finger Rotation) 포함한 전신 추정
- **Motivation:**
  - 정확한 3D Wrist Rotation에는 MCP 관절 정보 중요
  - 전신 Body Feature는 Hand 세부 정보 부족

#### 제안 방법
- **Overall Pipeline:** 
  - BodyNet: Body Parameters (θb, βb, Wrist Rotations)
  - HandNet: Hand Parameters (θh, Finger Rotations)
  - FaceNet: Face Parameters (θf, ψ)

- **HandNet 특징:**
  - **3D Finger Rotation:** Hand Feature만 사용 (Body Feature 사용 안함)
  - **3D Wrist Rotation:** Body Feature + MCP Hand Joint Feature 함께 사용
    - Body Feature: 손 비가시 상황에서 plausible 추정
    - MCP Feature: 정확한 Wrist Rotation 추정에 필수 정보

#### 성능 및 결과
- 다른 Whole-body 모델보다 손 부분이 자연스럽고 정확하게 추정
- 손(Hands) 부분 최저 에러 달성, 전체(All) 에러도 경쟁력 있는 성능

---

## 5. 3D Human Pose Estimation의 성과와 후속 연구

### 주요 성과
- **단일 RGB 이미지로부터 3D Pose Estimation:** 특수 장비(Depth Sensor, MoCap) 없이도 가능
- **3D Multi-Person Pose Estimation:** 여러 사람의 절대적 3D Pose 추정
- **3D Interacting Hand Pose Estimation:** 상호작용하는 두 손 추정 및 대규모 데이터셋 구축
- **비디오 기반 3D Pose Estimation:** 시간적 일관성 개선
- **3D Whole-Body Pose Estimation:** 전신 통합 추정 가능

### 산업계 기술 발전 현황
- **현재 상황:** 모바일/PC 조작이 주류, 완전한 3D 가상 아바타/상호작용은 아직 부족
- **가상 인간:** 주로 얼굴 교체나 미리 만든 3D 모델+모션캡쳐 기술 활용
- **Meta:** Metaverse 분야 선도 중, Oculus Quest 2의 성공으로 발전 가능성 보임

### 후속 연구 방향
- **Degraded Images/Videos:** 저해상도, 모션 블러 등 품질 낮은 영상에서의 3D 복원
- **Truncated Images/Videos:** 잘려나간 사람 이미지에서의 3D 복원
- **Hand Codec 개선:** 상호작용 손을 위한 Encoder 개선
- **Physical Interaction:** 물리적 상호작용 모델링
- **Personalizing Module:** 개인화된 3D 모델 추정

### 최종 목표
- **Seamless Human-Interacting Systems:** 인간-컴퓨터 상호작용(HCI)과 가상 3D 세계 구현
