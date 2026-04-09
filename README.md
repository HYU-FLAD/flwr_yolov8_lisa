# Flower + YOLOv8 + LISA federated training scaffold

이 템플릿은 다음 요구사항을 기준으로 만든 스캐폴드다.

- `flwr run .` 로 실행
- `pyproject.toml` 의 `[tool.flwr.app.config]` 를 통해 런타임 설정
- `YOLOv8n` 기본, `YOLOv8s` 로 손쉽게 변경 가능
- 기존 LISA 전처리 코드를 **브리지 함수 한 개**로 연결 가능
- 각 클라이언트가 TOML 기준으로 `is_attacker`, `attack_method` 를 확인 가능
- Ultralytics 기본 로그(`results.csv`, `best.pt`, `last.pt`) + 추가 FL 로그 저장
- 서버 측 라운드 로그(`server_history.csv`) + `last_global.pt`, `best_global.pt`, `final_global.pt`

## 1) 설치

```bash
pip install -e .
```

## 2) 실행

기본 로컬 GPU 시뮬레이션:

```bash
flwr run . --stream
```

CPU만 쓰고 싶으면:

```bash
flwr run . local --stream
```

YOLOv8s로 바꾸려면:

```bash
flwr run . --run-config "yolo-model='yolov8s.pt'"
```

## 3) LISA 전처리 연결

`pyproject.toml` 에 아래 값을 채우면 된다.

```toml
lisa-preprocess-function-path = "your_project.lisa_fl_bridge:prepare_partition"
lisa-accuracy-function-path = "your_project.lisa_fl_bridge:compute_accuracy"
```

### 전처리 함수 계약

함수 시그니처:

```python
def prepare_partition(
    partition_id: int,
    num_partitions: int,
    run_config: dict,
    output_dir: str,
) -> dict:
    ...
```

리턴 예시:

```python
return {
    "data_yaml": "/abs/path/to/client_0/data.yaml",
    "num_train_examples": 1450,
    "num_val_examples": 362,
    "metadata": {"split_seed": 42},
}
```

### accuracy 함수 계약

YOLO detect는 기본적으로 precision / recall / mAP를 기록한다. 일반적인 classification `acc` 스칼라는 기본 제공하지 않으므로,
기존 LISA 평가 코드에서 accuracy를 계산 중이라면 아래 브리지로 연결하면 `metrics/acc` 컬럼이 함께 기록된다.

```python
def compute_accuracy(
    weights_path: str,
    partition: dict,
    split: str,
    run_config: dict,
) -> float | None:
    ...
```

## 4) 공격자 로직

이 템플릿은 공격 코드를 넣지 않았다. 대신 아래 설정을 읽고 각 client trainer에서 확인한다.

```toml
attack-enabled = true
attack-method = "placeholder"
attacker-partitions = [1, 3]
attack-config-path = "configs/attack.toml"
```

현재 동작:

- 각 client는 자신의 `partition_id` 기준으로 attacker 여부 계산
- `federated_context.json` 에 공격 관련 설정 저장
- epoch 로그(`federated_metrics.jsonl/csv`)에 `is_attacker`, `attack_method` 기록
- trainer 내부 `NoOpAttackHook` 이 확장 포인트 역할 수행

## 5) 결과물 구조

예시:

```text
runs/flwr_yolov8_lisa/
├── prepared_data/
├── client_0/
│   ├── round_001/
│   │   ├── results.csv
│   │   ├── federated_metrics.csv
│   │   ├── federated_metrics.jsonl
│   │   ├── federated_context.json
│   │   ├── checkpoint_manifest.json
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── round_001_eval/
└── server/
    ├── server_history.csv
    ├── server_history.jsonl
    ├── run_summary.json
    └── weights/
        ├── best_global.pt
        ├── last_global.pt
        ├── final_global.pt
        └── round_001.pt
```

## 6) 파일 설명

- `client_app.py`: Flower `ClientApp`, train/evaluate 엔트리포인트
- `server_app.py`: Flower `ServerApp`, FedAvg 시작점
- `strategy.py`: 라운드 로그 + global checkpoint 저장 전략
- `trainer.py`: YOLOv8 custom trainer, FL 메타데이터 및 epoch 로그 저장
- `task.py`: 실제 로컬 train/eval 조립
- `lisa_bridge.py`: 기존 LISA 전처리 / accuracy 코드 연결용 얇은 어댑터
- `attack.py`: attacker 여부 판단 + no-op attack hook

## 7) 주의

- 이 템플릿은 **Ultralytics detection task** 기준이라 `acc` 는 optional이다.
- `attacker-partitions` 는 TOML에서는 리스트로 넣는 것이 가장 안전하다.
- Flower의 local simulation 자원 설정은 `pyproject.toml` 이 아니라 `.flwr/config.toml` 에 있다.
