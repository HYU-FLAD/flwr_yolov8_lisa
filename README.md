🌸 Federated YOLOv8 AnywhereDoor Attack
이 프로젝트는 연합학습(Federated Learning, FL) 환경에서 객체 탐지(Object Detection) 모델인 YOLOv8을 타겟으로 하는 고도화된 백도어 공격인 **'AnywhereDoor'**를 시뮬레이션하고 평가하는 프레임워크입니다. Flower(flwr) 1.12+ 엔진과 Ultralytics 프레임워크를 기반으로 구축되었습니다.

🚀 주요 기능 (Key Features)
학습 가능한 트리거 (Learnable Trigger): 고정된 이미지가 아닌, 모델의 취약점을 파고들도록 역전파(Backpropagation)를 통해 진화하는 텐서(Tensor) 기반 트리거 패치 생성.

교대 최적화 (2-Phase Alternating Optimization): 1. 모델 가중치 고정 & 트리거 최적화
2. 트리거 고정 & 모델 가중치 오염

알파 블렌딩 (Alpha Blending): 트리거를 원본 이미지에 반투명하게(Alpha=0.5) 합성하여 시각적 은밀성(Stealthiness) 확보.

SOTA 백도어 평가 (Strict ASR Evaluation): 글로벌 서버 평가 시 단순히 오탐지 여부만 확인하는 것이 아니라, Confidence ≥ 0.5 및 가짜 박스와 트리거 간의 IoU ≥ 0.3 조건을 통과해야만 공격 성공(ASR)으로 인정하는 엄격한 프로토콜 적용.

데이터 물리적 격리 (Symlink Data Isolation): FL 환경에서 여러 클라이언트가 YOLO 캐시(train.cache)를 공유하다 충돌하는 현상(Race Condition)을 막기 위해, 가상 링크 기반의 완벽한 샌드박스 데이터셋 자동 생성.

동적 학습률 제어 (Global LR Decay): 짧은 FL 로컬 에폭(3 Epoch)의 한계를 극복하기 위해, 서버 라운드에 비례하여 지수 감쇠(Exponential Decay)하는 학습률 스케줄러 자체 구현.

📁 디렉토리 구조 (Repository Structure)
Plaintext
flwr_yolov8_lisa_template/
├── pyproject.toml             # Flower 실행 환경, 하드웨어 할당 및 공격 파라미터 제어 센터
├── split.py                   # 캐시 충돌 방지용 가상 데이터셋(Symlink) 분할 스크립트
├── fl_yolo_backdoor/
│   ├── __init__.py
│   ├── client_app.py          # FL 클라이언트: YOLOv8 훈련, 동적 LR, 증강 제어
│   ├── server_app.py          # FL 서버: FedAvg 병합, 글로벌 mAP 및 ASR 평가, CSV 로깅
│   └── custom_trainer.py      # AnywhereDoor 백도어 공격 로직 (트리거 생성 및 교대 훈련)
└── datas/
    └── lisa_yolo/             # 원본 데이터셋 (go, stop, warning 3개 클래스)
⚙️ 요구 사항 (Prerequisites)
이 프로젝트는 다중 GPU 또는 다중 코어 CPU를 활용한 병렬 시뮬레이션(Ray)을 지원합니다.

Bash
pip install flwr>=1.12.0 ultralytics>=8.0.0 torch torchvision opencv-python
🎯 공격 파라미터 설정 (Configuration)
모든 시뮬레이션 및 백도어 공격 설정은 pyproject.toml 파일에서 중앙 집중식으로 관리됩니다.

Ini, TOML
[tool.flwr.app.config]
num-server-rounds = 50       # 총 연합학습 라운드 수
local-epochs = 3             # 클라이언트 당 로컬 학습 에폭
total-clients = 10           # 전체 클라이언트 수

# AnywhereDoor Backdoor Configurations
attack-flag = true           # 공격 활성화 여부 (true/false)
attacker-ratio = 0.2         # 전체 클라이언트 중 악의적 노드(스파이)의 비율 (예: 20%)
poison-rate = 0.5            # 공격자의 로컬 데이터 중 오염시킬 데이터의 비율
trigger-size = 32            # 트리거 패치 크기 (픽셀 단위, 예: 32x32)
target-class = 0             # 모델을 속여 인식하게 만들 목표 타겟 클래스 (0: go)
🏃‍♂️ 실행 방법 (How to Run)
Step 1: 데이터 분할 및 격리 환경 구성
연합학습을 시작하기 전, 반드시 아래 스크립트를 실행하여 클라이언트별 가상 데이터 폴더(client_isolated_N)를 생성해야 합니다.

Bash
python split.py
Step 2: 연합학습 시뮬레이션 시작
Flower 1.12+의 최신 실행 문법을 사용하여 pyproject.toml에 정의된 local-sim 환경(클라이언트당 CPU 4, GPU 1 할당)으로 시뮬레이션을 시작합니다.

Bash
flwr run .
📊 결과 로그 및 평가 지표 (Evaluation Metrics)
시뮬레이션이 진행됨에 따라 fl_logs/server_YYYYMMDD_HHMMSS/global_metrics.csv 파일이 생성되며, 매 라운드 다음 지표가 기록됩니다.

mAP50: 기본 탐지 성능 (Clean Accuracy 유지 여부 확인)

F1_Score / Precision / Recall: 정밀도 및 재현율 기반 척도

ASR (Attack Success Rate): 공격 성공률. 목표한 위치(트리거 부착점)에 타겟 클래스의 바운딩 박스가 정확히 생성되었는지를 측정합니다.

📝 References
AnywhereDoor: A Stealthy and Adaptive Backdoor Attack in Object Detection

Flower: A Friendly Federated Learning Framework

Ultralytics YOLOv8