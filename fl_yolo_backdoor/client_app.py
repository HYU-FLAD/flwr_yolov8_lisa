import os
import copy
import torch
import random
import glob
import cv2
import numpy as np
from pathlib import Path
import yaml
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from ultralytics import YOLO
from .custom_trainer import AnywhereDoorTrainer

FL_LOG_ROOT = os.environ.get("FL_LOG_ROOT", "/home/flba/project/flwr_yolov8_lisa_template/fl_logs")

def cfg_get(cfg, key, default):
    val = cfg.get(key, default)
    return default if val is None else val

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

class YOLOBadNetClient(NumPyClient):
    def __init__(self, data_path: str, node_id: int, is_attacker: bool, attack_config: dict):
        self.data_path = data_path
        self.node_id = node_id
        self.is_attacker = is_attacker
        self.attack_config = attack_config
        self.pid = os.getpid()
        
        self.client_dir = os.path.join(FL_LOG_ROOT, f"client_node{self.node_id}_pid{self.pid}")
        os.makedirs(self.client_dir, exist_ok=True)
        self.local_pt_path = os.path.join(self.client_dir, "global_model.pt")
        
        self.model = YOLO("yolov8n_custom.yaml")
        try: self.model.load("yolov8n.pt")
        except Exception: pass

    def get_parameters(self, config):
        return [val.cpu().numpy().astype(np.float32) for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys = list(self.model.model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v, dtype=self.model.model.state_dict()[k].dtype) for k, v in zip(keys, parameters)})
        self.model.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 1)
        
        AnywhereDoorTrainer.server_round = int(server_round)
        AnywhereDoorTrainer.is_attacker = self.is_attacker
        AnywhereDoorTrainer.attack_config = self.attack_config
        AnywhereDoorTrainer.pid = self.pid
        AnywhereDoorTrainer.client_dir = self.client_dir

        torch.save({"model": copy.deepcopy(self.model.model).float()}, self.local_pt_path)

        if self.is_attacker and self.attack_config.get("attack-flag", False):
            print(f"\n==============================================")
            print(f"😈 [ATTACKER Node-{self.node_id}] AnywhereDoor 학습 시작 (Round {server_round})")
            print(f"==============================================\n")

        overrides = dict(
            model=self.local_pt_path, data=self.data_path, epochs=int(config.get("local_epochs", 2)), 
            workers=0, batch=4, device=(self.node_id % torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'), 
            optimizer='SGD', save=True, plots=False, val=False, project=self.client_dir, name="train", exist_ok=True, 
            lr0=max(0.01 * (0.95 ** (server_round - 1)), 0.0001) 
        )
        
        trainer = AnywhereDoorTrainer(overrides=overrides)
        trainer.train()

        lbl_chg, poi_batch = 0, 0
        gen_path = os.path.join(self.client_dir, f"generator_round_{int(server_round)}.pt")
        
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            lbl_chg = getattr(trainer, 'debug_total_label_changed', 0)
            poi_batch = getattr(trainer, 'debug_total_poisoned_batches', 0)
            print(f"[DEBUG][CLIENT] G_phi saved: {os.path.exists(gen_path)} | Poisoned Batches: {poi_batch} | Label Flipped: {lbl_chg}")
        
        for pt_path in [os.path.join(self.client_dir, "train", "weights", "last.pt")]:
            if os.path.exists(pt_path):
                try: ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                except TypeError: ckpt = torch.load(pt_path, map_location="cpu")
                try: self.model.model.load_state_dict(ckpt["model"].float().state_dict(), strict=True)
                except: self.model.model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
                break
                
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        try: num_examples = len(trainer.train_loader.dataset)
        except Exception: num_examples = count_train_images(self.data_path)
            
        metrics = {
            "is_attacker": int(self.is_attacker),
            "label_changed": int(lbl_chg),
            "poisoned_batches": int(poi_batch),
            "node_id": str(self.node_id),
            "gen_path": str(gen_path)
            if (
                self.is_attacker
                and self.attack_config.get("attack-flag", False)
                and os.path.exists(gen_path)
            )
            else "",
        }

        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config): 
        return 0.0, 1, {}

def client_fn(context: Context):
    run_cfg = context.run_config
    tc = int(cfg_get(run_cfg, "total-clients", 100))
    nid = int(context.node_id) % tc
    
    cfg = {
        "seed": nid, 
        "attack-flag": cfg_get(run_cfg, "attack-flag", True),
        "attacker-ratio": cfg_get(run_cfg, "attacker-ratio", 0.5),
        "poison-rate": cfg_get(run_cfg, "poison-rate", 0.7),
        "trigger-size": cfg_get(run_cfg, "trigger-size", 32),
        "num-classes": cfg_get(run_cfg, "num-classes", 3),
        "epsilon": cfg_get(run_cfg, "epsilon", 0.10),
        "trigger-inner-steps": cfg_get(run_cfg, "trigger-inner-steps", 1),
        "eval-attack-mode": cfg_get(run_cfg, "eval-attack-mode", "targeted_miscls"),
        "generator-lr": cfg_get(run_cfg, "generator-lr", 0.01),
    }
    
    rng = random.Random(nid)
    is_attacker = rng.random() < float(cfg["attacker-ratio"])
    
    return YOLOBadNetClient(f"datas/lisa_yolo/client_isolated_{tc}/client_{nid}/data.yaml", nid, is_attacker, cfg).to_client()

app = ClientApp(client_fn=client_fn)