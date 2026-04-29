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

def count_train_images(data_yaml: str) -> int:
    try:
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)

        yaml_dir = Path(data_yaml).parent
        root = data_cfg.get("path", None)
        
        if root is not None:
            root = Path(root)
            if not root.is_absolute():
                root = yaml_dir / root
        else:
            root = yaml_dir

        train_path = Path(data_cfg["train"])
        if not train_path.is_absolute():
            train_path = root / train_path

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        count = sum(len(list(train_path.rglob(ext))) for ext in exts)
        return max(count, 1)
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
        
        self.client_dir = os.path.join(FL_LOG_ROOT, f"client_{self.pid}")
        os.makedirs(self.client_dir, exist_ok=True)
        self.local_pt_path = os.path.join(self.client_dir, "global_model.pt")
        
        self.model = YOLO("yolov8n_custom.yaml")
        try: self.model.load("yolov8n.pt")
        except Exception as e: pass
        self.model.model.eval()

    def get_parameters(self, config):
        return [val.cpu().numpy().astype(np.float32) for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys = list(self.model.model.state_dict().keys())
        state_dict = OrderedDict({
            k: torch.tensor(v, dtype=self.model.model.state_dict()[k].dtype) 
            for k, v in zip(keys, parameters)
        })
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
            print(f"😈 [ATTACKER Node-{self.node_id}] Targeted Miscls 시작 (Round {server_round})")
            print(f"==============================================\n")

        current_lr = max(0.01 * (0.95 ** (server_round - 1)), 0.0001)
        device_id = self.node_id % torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'

        overrides = dict(
            model=self.local_pt_path, data=self.data_path,
            epochs=int(config.get("local_epochs", 5)), workers=0, batch=4,                 
            device=device_id, optimizer='SGD', save=True, plots=False, val=False,
            project=self.client_dir, name="train", exist_ok=True, warmup_epochs=0.0, lr0=current_lr 
        )
        
        trainer = AnywhereDoorTrainer(overrides=overrides)
        trainer.train()

        lbl_chg = 0
        poi_batch = 0
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            lbl_chg = getattr(trainer, 'debug_total_label_changed', 0)
            poi_batch = getattr(trainer, 'debug_total_poisoned_batches', 0)
            print(f"[DEBUG][CLIENT] poison summary: selected(real)={getattr(trainer, 'debug_total_selected', 0)}, label_changed={lbl_chg}, poisoned_batches={poi_batch}")
            
            expected_trigger = os.path.join(self.client_dir, f"trigger_patch_round_{int(server_round)}.pt")
            trigger_exists = os.path.exists(expected_trigger)
            print(f"[DEBUG][CLIENT] trigger exists after train: {trigger_exists}")
        
        candidate_paths = [os.path.join(self.client_dir, "train", "weights", "last.pt"), os.path.join(self.client_dir, "train", "weights", "best.pt")]
        loaded = False
        for pt_path in candidate_paths:
            if os.path.exists(pt_path):
                try: ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                except TypeError: ckpt = torch.load(pt_path, map_location="cpu")
                
                ckpt_model = ckpt["model"].float()
                try: self.model.model.load_state_dict(ckpt_model.state_dict(), strict=True)
                except RuntimeError as e:
                    print(f"[WARN] strict=True 로드 실패. strict=False 적용: {e}")
                    result = self.model.model.load_state_dict(ckpt_model.state_dict(), strict=False)
                    print(f"[WARN] missing_keys={len(result.missing_keys)}, unexpected_keys={len(result.unexpected_keys)}")
                loaded = True
                print(f"[DEBUG] 가중치 로드 성공: {pt_path}")
                break
                
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        try: num_examples = len(trainer.train_loader.dataset)
        except Exception: num_examples = count_train_images(self.data_path)
            
        metrics = {
            "is_attacker": int(self.is_attacker),
            "label_changed": lbl_chg,
            "poisoned_batches": poi_batch
        }
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

def client_fn(context: Context):
    total_clients = context.run_config.get("total-clients", 100)
    node_id = context.node_id
    safe_node_id = node_id % total_clients
    data_path = f"datas/lisa_yolo/client_isolated_{total_clients}/client_{safe_node_id}/data.yaml"
    
    attack_config = {
        "attack-flag": context.run_config.get("attack-flag", False),
        "attacker-ratio": context.run_config.get("attacker-ratio", 0.0),
        "poison-rate": context.run_config.get("poison-rate", 0.0),
        "trigger-size": context.run_config.get("trigger-size", 32),
        "target-class": context.run_config.get("target-class", 0),
        "source-class": context.run_config.get("source-class", 1),
        "epsilon": context.run_config.get("epsilon", 0.05),
        "trigger-inner-steps": context.run_config.get("trigger-inner-steps", 1)
    }
    random.seed(safe_node_id) 
    is_attacker = random.random() < attack_config["attacker-ratio"]
    return YOLOBadNetClient(data_path, safe_node_id, is_attacker, attack_config).to_client()

app = ClientApp(client_fn=client_fn)