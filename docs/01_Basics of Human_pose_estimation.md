# Human Pose Estimation 강의 요약

## 목차
1. [Intro to Human Pose Estimation](#1-intro-to-human-pose-estimation)
2. [2D Human Pose Estimation](#2-2d-human-pose-estimation)
3. [3D Human Pose Estimation](#3-3d-human-pose-estimation)

---

## 1. Intro to Human Pose Estimation

### 컴퓨터 비전을 통한 인간 이해의 중요성
- 인간은 우리 삶에서 가장 중심적인 존재
- 예술(아누비스 신, 아테네 학당)이나 과학(레오나르도 다빈치의 비트루비우스적 인간) 분야에서 오랫동안 인간 형태 연구
- 현재는 시각 지능(Visual Intelligence)을 가진 기계를 통해 인간과 상호작용하는 연구 활발

### 산업 및 학계의 관심과 활용
- **주요 활용 분야:**
  - **모션 캡쳐:** 영화, 게임 캐릭터에 자연스러운 움직임 부여
  - **AR/VR:** 가상/증강 현실에서 아바타나 상호작용
  - **공공장소 감시:** 사람의 행동 분석
  - **Virtual try-on:** 옷을 가상으로 입어보는 서비스
  - **운동 보조 및 재활 치료:** 자세/움직임 분석하여 피드백 제공

- **주요 연구 기관/기업:** Facebook Reality Labs, Max Planck Institute, Naver Labs Europe, NVIDIA, Google, Microsoft 등

### Human Pose Estimation 정의
- 입력 이미지나 비디오로부터 사람의 관절(joints) 위치나 회전 정보를 추정하는 기술
- **정보 종류:** 2D/3D 관절 좌표 (x, y, (z)), 3D 관절 회전 등
- **Human-Understanding Computer Vision (HUCV)** 분야의 핵심 기술

### 다른 응용 연구들
- Person Re-Identification (사람 재식별)
- High-fidelity rendering
- Human action recognition (인간 행동 인식)
- Motion transfer
- Human image manipulation

### 입력 데이터 환경
- 일상 환경에서 촬영된 **"In the Wild"** 이미지에서 pose 추정이 중요
- 특히 **단일 이미지 기반(Single Image-Based)** Human Pose Estimation에 초점

---

## 2. 2D Human Pose Estimation

### 목표와 정의
- 입력 이미지로부터 사람의 관절을 **2D 공간에 위치화**
- **출력:** 각 사람의 각 관절에 대한 2D 좌표 (x, y) 목록
- **출력 형태:** NxJx2 (N: 사람 수, J: 관절 개수, 2: x,y 좌표)
  - 사람 수는 이미지마다 다르고, 관절 개수는 고정
  - 예: MSCOCO 데이터셋은 17개 관절 포함 ('Nose', 'L_Eye', ... 'R_Ankle' 등)

### 어려움
- **사람끼리의 가려짐 (Occlusion):** 여러 사람이 겹쳐서 일부 관절이 보이지 않음
- **복잡한 자세 (Complex poses):** 춤이나 격렬한 운동 등 예측이 어려운 자세
- **작은 해상도 (Low resolution):** 작게 나타난 사람의 관절 식별 어려움
- **모션 블러 (Motion blur):** 빠른 움직임으로 인한 이미지 흐림
- **잘림 (Truncation):** 이미지 경계에 걸쳐 일부만 보이는 경우

### 네트워크의 학습과 테스트
- **학습 (Training):**
  - **네트워크 구조:** 입력 이미지 → Feature Extractor → 2D Heatmap 출력
  - **Heatmap:** 각 관절별 확률 분포 맵 (JxHxW)
  - **Ground Truth Heatmap:** 각 관절 위치를 중심으로 **Gaussian blob** 형태로 생성
  - **Loss Function:** 추정된 heatmap과 Ground Truth heatmap 사이의 **L2 Loss**
  - **정의되지 않은 관절 처리:** 가려지거나 잘린 관절은 loss를 0으로 설정

- **테스트/추론 (Inference):**
  - 입력 이미지 → 학습된 네트워크 → 2D Heatmap 출력
  - **좌표 추출:** 각 관절 heatmap에 **argmax** 연산 적용하여 가장 확률이 높은 위치 추출

---

## 3. 3D Human Pose Estimation

### 목표와 정의
- 입력 이미지로부터 사람의 관절을 **3D 공간에 위치화**
- **3D Pose 정보:** 3D 관절 좌표 (x, y, z) 혹은 3D 관절 회전 정보
- **출력 형태:** NxJx3 (N: 사람 수, J: 관절 개수, 3: x,y,z 좌표)

### 어려움
- **2D와 공통된 어려움:** 가려짐, 복잡한 자세, 작은 해상도, 모션 블러, 잘림
- **3D 고유의 어려움:**
  - **Depth ambiguity (깊이 모호성):** 하나의 2D 이미지로부터 정확한 3D 정보 복원 어려움
  - **Data collection (데이터 수집):** 3D pose ground truth 데이터 수집 어려움
    - **주요 방법:** 모션 캡쳐 스튜디오, IMU 센서, LiDAR/TOF 스캔
    - **Domain Shift 문제:** MoCap 스튜디오 데이터와 실제 이미지 간 외형적 차이

### 네트워크의 학습과 테스트
- **학습 (Training):**
  - **네트워크 구조:** 입력 이미지 → Feature Extractor → 3D Pose 출력
  
  - **출력 방식 1: 3D 좌표 직접 예측**
    - 네트워크가 Jx3 형태의 3D 관절 좌표 직접 출력
    - **Loss Function:** 추정된 3D 좌표와 Ground Truth 3D 좌표 사이의 **L1 Loss**
    
  - **출력 방식 2: 3D 회전 예측 + Forward Kinematics (FK)**
    - 네트워크가 각 관절의 3D 회전 정보 예측
    - **Forward Kinematics (FK):** 인체 Kinematic chain 구조와 예측된 회전 정보로 3D 좌표 계산
    - **Loss Function:** FK를 통해 계산된 3D 좌표와 Ground Truth 3D 좌표 사이의 **L1 Loss**
  
  - **Root Joint-relative 3D Pose:**
    - 절대적 3D 위치 대신 특정 기준 관절(골반 등)에 대한 **상대적 3D 좌표** 사용
    - 깊이 모호성 영향 감소 및 학습 안정화

- **테스트/추론 (Inference):**
  - 입력 이미지 → 학습된 네트워크 → 3D Pose 출력
  - 회전 출력 시 FK로 최종 3D 관절 좌표 계산 (root joint-relative)
