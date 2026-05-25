import argparse
from pathlib import Path
import yaml


FIXED_FRACTION_FIT = 0.3
FIXED_TOTAL_CLIENTS = 100
FIXED_NUM_SERVER_ROUNDS = 100
FIXED_LOCAL_EPOCHS = 3

FIXED_ATTACKER_RATIO = 0.2
FIXED_POISON_RATE = 0.3


def parse_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["clean", "attack"], required=True)
    parser.add_argument("--method", default="AnywhereDoor")

    parser.add_argument("--data-yaml", default="datas/lisa_yolo_full/data.yaml")
    parser.add_argument("--partition-root", default="datas/lisa_yolo_full/client_isolated_100")
    parser.add_argument("--model-yaml", default="yolov8n_lisa_full.yaml")
    parser.add_argument("--pretrained", default="yolov8n.pt")

    parser.add_argument("--source-name", default="stop")
    parser.add_argument("--target-name", default="go")

    # Method-specific attack strength options.
    # These are NOT tuned by results. Set once according to the paper/protocol.
    parser.add_argument("--trigger-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--generator-lr", type=float, default=0.01)
    parser.add_argument("--trigger-inner-steps", type=int, default=1)

    # Fixed evaluation options for reproducible table-up.
    parser.add_argument("--asr-conf-thresh", type=float, default=0.1)
    parser.add_argument("--asr-num-samples", type=int, default=500)
    parser.add_argument("--asr-seed", type=int, default=2026)

    parser.add_argument("--out", default="pyproject.toml")

    args = parser.parse_args()

    data = yaml.safe_load(Path(args.data_yaml).read_text())
    names = data["names"]
    nc = int(data["nc"])

    if args.source_name not in names:
        print("[ERROR] source-name not found in names.")
        print("Available names:")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        raise SystemExit(1)

    if args.target_name not in names:
        print("[ERROR] target-name not found in names.")
        print("Available names:")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        raise SystemExit(1)

    src_id = names.index(args.source_name)
    tgt_id = names.index(args.target_name)

    if args.mode == "clean":
        attack_flag = False
        attacker_ratio = 0.0
        poison_rate = 0.0
    else:
        attack_flag = True
        attacker_ratio = FIXED_ATTACKER_RATIO
        poison_rate = FIXED_POISON_RATE

    toml = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fl-yolo-backdoor"
version = "3.11.0"
description = "Federated Learning YOLO Backdoor Fixed Protocol"
license = "MIT"
dependencies = [
    "flwr>=1.12.0",
    "ultralytics>=8.0.0",
    "torch>=2.0.0",
    "torchvision",
    "opencv-python",
    "pyyaml"
]

[tool.flwr.app]
publisher = "limulu"

[tool.flwr.app.components]
serverapp = "fl_yolo_backdoor.server_app:app"
clientapp = "fl_yolo_backdoor.client_app:app"

[tool.flwr.app.config]
num-server-rounds = {FIXED_NUM_SERVER_ROUNDS}
local-epochs = {FIXED_LOCAL_EPOCHS}
total-clients = {FIXED_TOTAL_CLIENTS}
fraction-fit = {FIXED_FRACTION_FIT}

dataset-name = "lisa_full"
num-classes = {nc}

data-yaml = "{args.data_yaml}"
partition-root = "{args.partition_root}"
model-yaml = "{args.model_yaml}"
pretrained-weights = "{args.pretrained}"

method-name = "{args.method}"

reset-global-generator = true

attack-flag = {str(attack_flag).lower()}
attacker-ratio = {attacker_ratio}
poison-rate = {poison_rate}
trigger-size = {args.trigger_size}

eval-attack-mode = "targeted_miscls"

fixed-source-class = {src_id}
fixed-target-class = {tgt_id}

epsilon = {args.epsilon}
generator-lr = {args.generator_lr}
trigger-inner-steps = {args.trigger_inner_steps}

asr-conf-thresh = {args.asr_conf_thresh}
asr-max-pairs = 6
asr-num-samples = {args.asr_num_samples}
asr-seed = {args.asr_seed}
'''

    Path(args.out).write_text(toml)

    print(f"[DONE] Wrote {args.out}")
    print(f"[INFO] mode = {args.mode}")
    print(f"[INFO] method = {args.method}")
    print(f"[INFO] nc = {nc}")
    print(f"[INFO] source = {args.source_name} -> {src_id}")
    print(f"[INFO] target = {args.target_name} -> {tgt_id}")
    print(f"[INFO] fraction_fit = {FIXED_FRACTION_FIT}")
    print(f"[INFO] total_clients = {FIXED_TOTAL_CLIENTS}")
    print(f"[INFO] num_server_rounds = {FIXED_NUM_SERVER_ROUNDS}")
    print(f"[INFO] local_epochs = {FIXED_LOCAL_EPOCHS}")
    print(f"[INFO] attack_flag = {attack_flag}")
    print(f"[INFO] attacker_ratio = {attacker_ratio}")
    print(f"[INFO] poison_rate = {poison_rate}")


if __name__ == "__main__":
    main()
