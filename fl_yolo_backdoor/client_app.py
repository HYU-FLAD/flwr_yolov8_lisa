import os
import copy
import torch
import random
import numpy as np
import json
from pathlib import Path
import yaml
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from ultralytics import YOLO
from .custom_trainer import AnywhereDoorTrainer

FL_LOG_ROOT = os.environ.get("FL_LOG_ROOT", "/home/flba/project/flwr_yolov8_lisa_template/fl_logs_nc7")

def cfg_get(cfg, key, default):
    val = cfg.get(key, default)
    return default if val is None else val

def parse_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ["1", "true", "yes", "y", "on"]
    return bool(x)

def resolve_client_data_path(run_cfg, nid: int, tc: int) -> str:
    partition_root = cfg_get(run_cfg, "partition-root", None)
    if partition_root:
        return os.path.abspath(str(Path(partition_root) / f"client_{nid}" / "data.yaml"))

    data_yaml = cfg_get(run_cfg, "data-yaml", None)
    if data_yaml:
        return os.path.abspath(str(data_yaml))

    raise ValueError("[FATAL] partition-root or data-yaml must be set in pyproject.toml")

def count_train_images(data_yaml: str) -> int:
    try:
        with open(data_yaml, "r") as f: data_cfg = yaml.safe_load(f)
        yaml_dir = Path(data_yaml).parent
        root = Path(data_cfg.get("path", yaml_dir))
        if not root.is_absolute(): root = yaml_dir / root
        train_path = Path(data_cfg["train"])
        if not train_path.is_absolute(): train_path = root / train_path
        return max(sum(len(list(train_path.rglob(ext))) for ext in ["*.jpg", "*.jpeg", "*.png"]), 1)
    except Exception as e:
        print(f"[WARN] count_train_images 실패. 기본값 1 반환: {e}")
        return 1

def state_l2_delta(before, after) -> float:
    total = 0.0
    for k in before.keys():
        if k in after and before[k].shape == after[k].shape:
            total += torch.sum((before[k].float() - after[k].float()) ** 2).item()
    return float(np.sqrt(total))

def resolve_client_id(context: Context, total_clients: int) -> int:
    node_cfg = getattr(context, "node_config", {}) or {}
    for key in ["partition-id", "partition_id", "cid", "client_id"]:
        if key in node_cfg:
            nid = int(node_cfg[key])
            if not (0 <= nid < total_clients):
                raise ValueError(f"[FATAL] {key}={nid} outside [0,{total_clients-1}]")
            return nid
    nid = int(context.node_id) % total_clients
    return nid

def assert_model_nc(yolo_obj, expected_nc: int, where: str):
    actual_nc = int(getattr(yolo_obj.model.model[-1], "nc", -1))
    if actual_nc != expected_nc:
        raise ValueError(
            f"[{where}] Model nc mismatch: actual_nc={actual_nc}, expected_nc={expected_nc}"
        )

def assert_data_yaml_nc(data_yaml: str, expected_nc: int, where: str):
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    actual_nc = int(data_cfg.get("nc", -1))
    names = data_cfg.get("names", None)

    if actual_nc != expected_nc:
        raise ValueError(
            f"[{where}] data.yaml nc mismatch: "
            f"data_nc={actual_nc}, expected_nc={expected_nc}, path={data_yaml}"
        )

    if names is not None and len(names) != expected_nc:
        raise ValueError(
            f"[{where}] data.yaml names length mismatch: "
            f"len(names)={len(names)}, expected_nc={expected_nc}, path={data_yaml}"
        )

class FlowerYoloClient(NumPyClient):
    def __init__(self, nid: int, is_attacker: bool, attack_config: dict, run_config: dict, tc: int):
        self.nid = nid
        self.is_attacker = is_attacker
        self.attack_config = attack_config
        self.run_config = run_config
        self.tc = tc
        
        self.pid = os.getpid()
        self.client_dir = os.path.join(FL_LOG_ROOT, f"client_{self.nid}")
        os.makedirs(self.client_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.client_dir, "local_model.pt")
        self.data_yaml = resolve_client_data_path(self.run_config, self.nid, self.tc)
        
        self.model_yaml = str(cfg_get(run_config, "model-yaml", "yolov8n_custom.yaml"))
        self.pretrained_weights = str(cfg_get(run_config, "pretrained-weights", "yolov8n.pt"))
        
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"[FATAL] Dataset YAML not found: {self.data_yaml}")
            
        assert_data_yaml_nc(self.data_yaml, int(self.attack_config["num-classes"]), f"client-{self.nid}")

    def get_parameters(self, config):
        expected_nc = int(self.attack_config["num-classes"])
        
        m = YOLO(self.model_yaml)
        if self.pretrained_weights.lower() not in ["none", ""]:
            try:
                m.load(self.pretrained_weights)
            except Exception as e:
                print(f"[WARN][CLIENT-{self.nid}] pretrained load failed in get_parameters: {e}")
                
        assert_model_nc(m, expected_nc, f"client-{self.nid}-get_parameters")
        
        return [val.detach().cpu().numpy().astype(np.float32) for val in m.model.state_dict().values()]

    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)
        
        attack_start_round = int(self.attack_config.get("attack-start-round", 5))
        effective_attack_flag = (
            bool(self.attack_config.get("attack-flag", False))
            and int(server_round) >= attack_start_round
        )
        
        round_attack_config = dict(self.attack_config)
        round_attack_config["attack-flag"] = effective_attack_flag

        base_model = YOLO(self.model_yaml)
        if self.pretrained_weights.lower() not in ["none", ""]:
            try:
                base_model.load(self.pretrained_weights)
            except Exception:
                pass

        assert_model_nc(base_model, int(self.attack_config["num-classes"]), f"client-{self.nid}")

        state = base_model.model.state_dict()
        
        if len(parameters) != len(state):
            raise ValueError(
                f"[FATAL][CLIENT] parameter length mismatch: "
                f"received={len(parameters)}, expected={len(state)}"
            )

        new_state = OrderedDict()
        for k, v in zip(state.keys(), parameters):
            tensor_v = torch.tensor(v, dtype=state[k].dtype)
            if tuple(tensor_v.shape) != tuple(state[k].shape):
                raise ValueError(
                    f"[FATAL][CLIENT] shape mismatch for {k}: "
                    f"received={tuple(tensor_v.shape)}, expected={tuple(state[k].shape)}"
                )
            new_state[k] = tensor_v

        base_model.model.load_state_dict(new_state, strict=True)
        
        torch.save({"model": copy.deepcopy(base_model.model).float()}, self.model_path)
        model = YOLO(self.model_path)
        
        before_state = copy.deepcopy(model.model.state_dict())
        
        epochs = int(self.run_config.get("local-epochs", 3))
        batch_size = int(self.run_config.get("local-batch", 8))
        imgsz = int(self.run_config.get("yolo-imgsz", 640))
        
        device_id = 0 if torch.cuda.is_available() else "cpu"
        print(
            f"[DEBUG][DEVICE] nid={self.nid}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"cuda_count={torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        overrides = {
            "model": self.model_path,
            "data": self.data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": device_id,
            "workers": 0,
            "project": os.path.join(self.client_dir, "runs"),
            "name": f"train_round_{server_round}",
            "exist_ok": True,
            "plots": False,
            "save": True,
            "verbose": False,
            "val": False,
            "optimizer": self.run_config.get("optimizer", "AdamW"),
            "lr0": float(self.run_config.get("local-lr0", 0.001)),
            "lrf": float(self.run_config.get("min-lr", 0.0001)) / float(self.run_config.get("local-lr0", 0.001)),
            "mosaic": float(self.run_config.get("yolo-mosaic", 0.0)),
            "erasing": float(self.run_config.get("yolo-erasing", 0.0)),
            "fliplr": float(self.run_config.get("yolo-fliplr", 0.0)),
            "hsv_h": float(self.run_config.get("yolo-hsv-h", 0.003)),
            "hsv_s": float(self.run_config.get("yolo-hsv-s", 0.1)),
            "hsv_v": float(self.run_config.get("yolo-hsv-v", 0.1)),
            "scale": float(self.run_config.get("yolo-scale", 0.1)),
            "translate": float(self.run_config.get("yolo-translate", 0.03)),
            "warmup_epochs": float(self.run_config.get("yolo-warmup-epochs", 0.0)),
        }
        
        freeze_val = self.run_config.get("yolo-freeze", None)
        if freeze_val is not None:
            overrides["freeze"] = int(freeze_val)
        
        custom_args = {
            "server_round": server_round,
            "is_attacker": self.is_attacker,
            "attack_config": round_attack_config,
            "pid": self.pid,
            "client_dir": self.client_dir,
        }
        
        if self.is_attacker and effective_attack_flag:
            print(f"😈 [ATTACKER Node-{self.nid}] AnywhereDoor 학습 시작 (Round {server_round})")
        elif self.is_attacker and not effective_attack_flag:
            print(f"😇 [ATTACKER Node-{self.nid}] 공격 대기 중 (Warm-up 구간 - Round {server_round})")
        
        trainer = AnywhereDoorTrainer(
            overrides=overrides,
            custom_args=custom_args
        )
        trainer.train()
        
        last_pt = os.path.join(
            self.client_dir,
            "runs",
            f"train_round_{server_round}",
            "weights",
            "last.pt",
        )
        if not os.path.exists(last_pt):
            raise FileNotFoundError(f"[FATAL] Checkpoint not found: {last_pt}")

        ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
        model.model.load_state_dict(ckpt["model"].float().state_dict(), strict=True)
        torch.save({"model": copy.deepcopy(model.model).float()}, self.model_path)
        
        after_state = model.model.state_dict()
        l2_dist = state_l2_delta(before_state, after_state)
        num_imgs = count_train_images(self.data_yaml)
        
        gen_path = os.path.join(self.client_dir, f"generator_round_{server_round}.pt")
        
        metrics = {
            "nid": int(self.nid),
            "pid": int(self.pid),
            "is_attacker": int(self.is_attacker and effective_attack_flag),
            "param_delta": float(l2_dist),
            "gen_path": str(gen_path) if (
                self.is_attacker and effective_attack_flag and os.path.exists(gen_path)
            ) else "",
            "label_changed": int(getattr(trainer, "debug_total_label_changed", 0)),
            "poisoned_batches": int(getattr(trainer, "debug_total_poisoned_batches", 0)),
            "gen_loss_labels": int(getattr(trainer, "debug_total_gen_loss_labels", 0)),
            "gen_steps": int(getattr(trainer, "debug_total_gen_steps", 0)),
        }
        
        updated_params = [
            val.detach().cpu().numpy().astype(np.float32)
            for val in after_state.values()
        ]
        return updated_params, num_imgs, metrics

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

def client_fn(context: Context):
    run_cfg = context.run_config
    tc = int(cfg_get(run_cfg, "total-clients", 100))
    
    nid = resolve_client_id(context, tc)
    
    attacker_ids_path = cfg_get(run_cfg, "attacker-ids-json", None)
    if attacker_ids_path and os.path.exists(attacker_ids_path):
        with open(attacker_ids_path, "r") as f:
            attacker_ids = set(json.load(f).get("attackers", []))
        is_attacker = nid in attacker_ids
    else:
        rng = random.Random(nid)
        is_attacker = rng.random() < float(cfg_get(run_cfg, "attacker-ratio", 0.5))
    
    fixed_src = cfg_get(run_cfg, "fixed-source-class", None)
    fixed_tgt = cfg_get(run_cfg, "fixed-target-class", None)

    attack_cfg = {
        "seed": int(nid),
        "attack-flag": parse_bool(cfg_get(run_cfg, "attack-flag", False)),
        "attack-start-round": int(cfg_get(run_cfg, "attack-start-round", 5)),
        "attacker-ratio": float(cfg_get(run_cfg, "attacker-ratio", 0.5)),
        "poison-rate": float(cfg_get(run_cfg, "poison-rate", 0.0)),
        "trigger-size": int(cfg_get(run_cfg, "trigger-size", 32)),
        "num-classes": int(cfg_get(run_cfg, "num-classes", 7)),
        "epsilon": float(cfg_get(run_cfg, "epsilon", 0.10)),
        "generator-lr": float(cfg_get(run_cfg, "generator-lr", 0.01)),
        "trigger-inner-steps": int(cfg_get(run_cfg, "trigger-inner-steps", 1)),
        "generator-target-only": parse_bool(cfg_get(run_cfg, "generator-target-only", False)),
        "fixed-source-class": None if fixed_src is None else int(fixed_src),
        "fixed-target-class": None if fixed_tgt is None else int(fixed_tgt),
        "eval-attack-mode": str(cfg_get(run_cfg, "eval-attack-mode", "targeted_miscls")),
    }
    
    print(
        f"[DEBUG][CFG] nid={nid}, "
        f"attack_flag={attack_cfg['attack-flag']}, "
        f"attack_start_round={attack_cfg['attack-start-round']}, "
        f"attacker_ratio={attack_cfg['attacker-ratio']}, "
        f"poison_rate={attack_cfg['poison-rate']}, "
        f"fixed_src={attack_cfg.get('fixed-source-class')}, "
        f"fixed_tgt={attack_cfg.get('fixed-target-class')}"
    )
    
    return FlowerYoloClient(
        nid=nid,
        is_attacker=is_attacker,
        attack_config=attack_cfg,
        run_config=run_cfg,
        tc=tc
    ).to_client()

app = ClientApp(client_fn=client_fn)