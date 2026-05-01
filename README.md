# FL-YOLO Backdoor: Federated YOLOv8 with AnywhereDoor-style Adaptive Trigger Attack

Flower 기반 Federated Learning 환경에서 YOLOv8 객체 탐지 모델을 학습하고, LISA traffic light dataset에 대해 AnywhereDoor-style adaptive trigger backdoor attack을 실험하기 위한 프로젝트입니다.

현재 구현은 **YOLOv8 + Flower + LISA dataset + generator-based adaptive trigger** 구조를 사용합니다. 공격은 `G_phi` generator를 이용해 trigger noise를 생성하고, 공격 client에서 source class label을 target class label로 변경하는 targeted misclassification 형태로 동작합니다.

> 현재 코드 기준 주의사항  
> 이 구현은 논문 AnywhereDoor의 전체 multi-target/multi-object setting을 완전히 재현한 구현이라기보다는, **고정 source-target class pair 기반의 single-pair targeted misclassification 실험**에 가깝습니다.  
> 기본 설정은 `fixed-source-class = 0`, `fixed-target-class = 1`입니다.

---

## 1. Project Overview

이 프로젝트의 목적은 다음과 같습니다.

- Flower 기반 Federated Learning 환경에서 YOLOv8 객체 탐지 모델 학습
- LISA traffic light dataset을 client partition 형태로 분할하여 FL 학습 수행
- 공격 client에서 adaptive trigger generator `G_phi`를 이용한 backdoor poisoning 수행
- 서버 측 global evaluation으로 매 round마다 clean mAP50, Precision, Recall, F1, ASR 기록
- Round별 global metrics를 CSV로 저장
- 공격 client의 label flip 수와 attacker 참여 수를 기록

---

## 2. Main Features

### Federated Learning

- Flower `ServerApp`, `ClientApp` 기반 구조
- `flwr run .` 방식 실행 지원
- `pyproject.toml` 기반 실험 설정 관리
- FedAvg aggregation 사용
- client별 local YOLO training 수행
- 서버에서 round별 global model evaluation 수행

### YOLOv8 Object Detection

- Ultralytics YOLOv8 사용
- custom 3-class detector 설정
- LISA dataset 기준 class 구성
  - `go`
  - `stop`
  - `warning`

### AnywhereDoor-style Backdoor Attack

- attacker client에서만 poisoning 수행
- `AnywhereDoorGenerator`를 통해 trigger pattern 생성
- source class와 target class one-hot vector를 generator 입력으로 사용
- image 전체에 mosaicked trigger noise 적용
- targeted misclassification 방식으로 label 변경
- generator `G_phi`를 각 attacker client에서 local update 후 서버에서 평균 aggregation

### Metrics Logging

서버는 매 round마다 다음 지표를 CSV로 저장합니다.

- `Round`
- `mAP50`
- `F1_Score`
- `Precision`
- `Recall`
- `ASR_Miscls_Avg`
- `ASR_Removal_Avg`
- `Active_ASR`
- `Miscls_Targets`
- `Removal_Targets`
- `Num_Attackers`
- `Total_Label_Changed`

---

## 3. Repository Structure

예상 프로젝트 구조는 다음과 같습니다.

```text
flwr_yolov8_lisa_template/
├── fl_yolo_backdoor/
│   ├── __init__.py
│   ├── client_app.py
│   ├── server_app.py
│   └── custom_trainer.py
├── datas/
│   └── lisa_yolo/
│       ├── data.yaml
│       ├── images/
│       ├── labels/
│       └── client_isolated_100/
│           ├── client_0/
│           │   └── data.yaml
│           ├── client_1/
│           │   └── data.yaml
│           └── ...
├── pyproject.toml
├── yolov8n_custom.yaml
├── README.md
└── fl_logs/
```

### Core files

| File                                                        | Description                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| `pyproject.toml`                                            | Flower app 설정 및 실험 hyperparameter 관리                  |
| `fl_yolo_backdoor/client_app.py`                            | Flower client 구현, local YOLO 학습, attacker client 설정    |
| `fl_yolo_backdoor/server_app.py`                            | Flower server 구현, FedAvg, global evaluation, ASR 측정, CSV 저장 |
| `fl_yolo_backdoor/custom_trainer.py`                        | YOLO custom trainer, adaptive trigger injection, generator optimization |
| `yolov8n_custom.yaml`                                       | YOLOv8 custom 3-class model config                           |
| `datas/lisa_yolo/data.yaml`                                 | global validation dataset config                             |
| `datas/lisa_yolo/client_isolated_100/client_{id}/data.yaml` | client별 local dataset config                                |

---

## 4. Environment

현재 실행 로그 기준 환경 예시는 다음과 같습니다.

```text
Python       3.10
Flower       >= 1.12.0
Ultralytics  8.4.14
PyTorch      2.5.1+cu121
CUDA         12.1
GPU          NVIDIA GeForce RTX 2080 Ti
```

`pyproject.toml`의 dependency 예시는 다음과 같습니다.

```toml
dependencies = [
    "flwr>=1.12.0",
    "ultralytics>=8.0.0",
    "torch>=2.0.0",
    "torchvision",
    "opencv-python",
    "pyyaml"
]
```

---

## 5. Installation

Conda 환경 예시입니다.

```bash
conda create -n fl_yolo8 python=3.10 -y
conda activate fl_yolo8
```

필요 패키지를 설치합니다.

```bash
pip install -U pip
pip install flwr ultralytics torch torchvision opencv-python pyyaml
```

프로젝트 루트에서 editable install이 필요하면 다음을 실행합니다.

```bash
pip install -e .
```

---

## 6. Dataset Preparation

이 프로젝트는 LISA traffic light dataset을 YOLO format으로 변환한 구조를 전제로 합니다.

기본 global validation config는 다음 경로를 사용합니다.

```text
datas/lisa_yolo/data.yaml
```

client별 partition은 다음 구조를 기대합니다.

```text
datas/lisa_yolo/client_isolated_100/client_0/data.yaml
datas/lisa_yolo/client_isolated_100/client_1/data.yaml
...
datas/lisa_yolo/client_isolated_100/client_99/data.yaml
```

각 client의 `data.yaml`은 해당 client의 local train set을 가리켜야 하며, global validation은 서버 평가에서 `datas/lisa_yolo/data.yaml`을 사용합니다.

---

## 7. Configuration

주요 설정은 `pyproject.toml`의 `[tool.flwr.app.config]`에 정의합니다.

현재 실험 설정 예시는 다음과 같습니다.

```toml
[tool.flwr.app.config]
num-server-rounds = 50
local-epochs = 2
total-clients = 100
fraction-fit = 0.2

reset-global-generator = true

attack-flag = true
attacker-ratio = 0.5
poison-rate = 0.5
trigger-size = 32
num-classes = 3

eval-attack-mode = "targeted_miscls"

fixed-source-class = 0
fixed-target-class = 1

epsilon = 0.10
generator-lr = 0.01
trigger-inner-steps = 1

asr-conf-thresh = 0.1
asr-max-pairs = 6
asr-num-samples = 100
asr-seed = 2026
```

### Important Parameters

| Parameter             | Meaning                                             |
| --------------------- | --------------------------------------------------- |
| `num-server-rounds`   | FL server aggregation round 수                      |
| `local-epochs`        | client local training epoch 수                      |
| `total-clients`       | 전체 client 수                                      |
| `fraction-fit`        | round마다 fit에 참여할 client 비율                  |
| `attack-flag`         | backdoor attack 활성화 여부                         |
| `attacker-ratio`      | 전체 client 중 attacker로 동작할 확률               |
| `poison-rate`         | attacker client 내부에서 image/batch poisoning 확률 |
| `trigger-size`        | trigger patch 크기                                  |
| `epsilon`             | image에 추가되는 trigger noise 강도                 |
| `generator-lr`        | trigger generator optimizer learning rate           |
| `trigger-inner-steps` | batch마다 generator를 최적화하는 inner step 수      |
| `fixed-source-class`  | 공격 source class                                   |
| `fixed-target-class`  | 공격 target class                                   |
| `eval-attack-mode`    | 현재 `targeted_miscls` 사용                         |
| `asr-conf-thresh`     | ASR 측정 시 YOLO prediction confidence threshold    |
| `asr-num-samples`     | ASR 측정에 사용할 validation image sample 수        |

---

## 8. Running Experiments

프로젝트 루트에서 다음 명령으로 실행합니다.

```bash
flwr run . --stream
```

설정을 command line에서 덮어쓰려면 다음과 같이 실행할 수 있습니다.

```bash
flwr run . --stream --run-config "num-server-rounds=10 local-epochs=2 fraction-fit=0.2 attack-flag=true attacker-ratio=0.5 poison-rate=0.5 trigger-size=32"
```

---

## 9. Output and Logs

기본 로그 디렉터리는 다음 경로입니다.

```text
/home/flba/project/flwr_yolov8_lisa_template/fl_logs
```

환경변수로 변경할 수 있습니다.

```bash
export FL_LOG_ROOT=/path/to/fl_logs
```

### Server logs

서버는 실행 시점별 디렉터리를 생성합니다.

```text
fl_logs/server_YYYYMMDD_HHMMSS/
└── global_metrics.csv
```

### Client logs

각 client는 node id와 process id 기반 디렉터리를 생성합니다.

```text
fl_logs/client_node{node_id}_pid{pid}/
├── global_model.pt
├── generator_round_{round}.pt
└── train/
    └── weights/
        ├── last.pt
        └── best.pt
```

---

## 10. Global Metrics CSV Naming Rule

실험 결과 CSV는 다음 규칙으로 정리할 수 있습니다.

```text
{version}_{clients&fraction}_{attack_method}_{poison_rate}_{patch_size}.csv
```

현재 설정 기준 예시는 다음과 같습니다.

```text
3.6.0_10020_AnywhereDoor_pr0.5_32.csv
```

구성은 다음과 같습니다.

| Component        | Value                                         |
| ---------------- | --------------------------------------------- |
| version          | `3.6.0`                                       |
| clients&fraction | `10020` = total clients 100, fraction-fit 0.2 |
| attack_method    | `AnywhereDoor`                                |
| poison_rate      | `pr0.5`                                       |
| patch_size       | `32`                                          |

예시 rename command:

```bash
mv global_metrics.csv 3.6.0_10020_AnywhereDoor_pr0.5_32.csv
```

---

## 11. Attack Logic

### 11.1 Generator

`AnywhereDoorGenerator`는 source class와 target class one-hot vector를 입력받아 trigger patch를 생성합니다.

```text
G_phi(e_r, e_g) -> trigger patch
```

현재 구현에서는 두 개의 branch를 사용합니다.

- `G_r`: source/removal branch
- `G_g`: target/misclassification branch

출력 trigger는 image 전체에 tile 형태로 반복 적용됩니다.

### 11.2 Poisoning

attacker client의 training batch에서 다음 조건을 만족하면 poisoning을 수행합니다.

- 현재 client가 attacker임
- `attack-flag = true`
- image 내에 `fixed-source-class` 객체가 존재함
- random sampling이 `poison-rate` 조건을 만족함

targeted misclassification의 경우 source class label을 target class label로 변경합니다.

```text
source class 0 -> target class 1
```

현재 기본 class mapping 기준으로는 다음 공격입니다.

```text
go -> stop
```

### 11.3 Generator Optimization

각 poisoned batch에 대해 generator는 inner optimization을 수행합니다.

```text
trigger-inner-steps = 1
```

동작 개요:

1. batch 복제
2. trigger injection
3. detector model parameter freeze
4. generator만 gradient update
5. detach된 trigger로 최종 poisoned batch 생성
6. detector local training 진행

---

## 12. ASR Evaluation

서버는 global model 평가 후 다음 순서로 ASR을 측정합니다.

1. round별 attacker client들이 저장한 `generator_round_{round}.pt` 수집
2. generator weight 평균화
3. global `G_phi` 저장
4. validation images sampling
5. clean image prediction 수행
6. source class prediction이 존재하는 box를 기준으로 target class 전환 여부 측정

현재 ASR은 GT annotation 기준이 아니라 **clean prediction 기준**입니다.

따라서 초반 round에서 model이 source class를 거의 탐지하지 못하면 ASR 분모가 0이 되어 ASR이 0으로 기록될 수 있습니다.

