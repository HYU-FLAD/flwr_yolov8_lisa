from pathlib import Path
from typing import Any

def prepare_partition(
    partition_id: int,
    num_partitions: int,
    run_config: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    data_yaml = Path("datas/lisa_yolo/data.yaml").resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(f"Shared data.yaml not found: {data_yaml}")

    return {
        "data_yaml": str(data_yaml),
        "num_train_examples": 0,
        "num_val_examples": 0,
        "metadata": {
            "partition_id": partition_id,
            "num_partitions": num_partitions,
            "mode": "shared_yaml_smoke_test",
        },
    }