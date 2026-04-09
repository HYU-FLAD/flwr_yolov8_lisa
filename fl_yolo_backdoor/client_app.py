import os
import copy
import torch
import random
import glob
import cv2
import numpy as np
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from ultralytics import YOLO
from .custom_trainer import AnywhereDoorTrainer

class YOLOBadNetClient(NumPyClient):
    def __init__(self, data_path: str, node_id: int, is_attacker: bool, attack_config: dict):
        self.data_path = data_path
        self.node_id = node_id
        self.is_attacker = is_attacker
        self.attack_config = attack_config
        self.pid = os.getpid()
        
        self.client_dir = f"fl_logs/client_{self.pid}"
        os.makedirs(self.client_dir, exist_ok=True)
        self.local_pt_path = os.path.join(self.client_dir, "global_model.pt")
        
        self.model = YOLO("yolov8n_custom.yaml")
        try:
            self.model.load("yolov8n.pt")
        except Exception as e:
            pass
        
        self.model.model.eval()
        dummy_input = torch.zeros((1, 3, 640, 640))
        _ = self.model.model(dummy_input)
        
        AnywhereDoorTrainer.pid = self.pid
        AnywhereDoorTrainer.client_dir = self.client_dir
        self.model.task_map["detect"]["trainer"] = AnywhereDoorTrainer

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
        
        torch.save({"model": copy.deepcopy(self.model.model).float()}, self.local_pt_path)
        train_model = YOLO(self.local_pt_path)
        train_model.task_map["detect"]["trainer"] = AnywhereDoorTrainer
        
        AnywhereDoorTrainer.is_attacker = self.is_attacker
        AnywhereDoorTrainer.attack_config = self.attack_config

        if self.is_attacker and self.attack_config.get("attack-flag", False):
            print(f"😈 [ATTACKER Node-{self.node_id}] AnywhereDoor 공격 시작")

        server_round = config.get("server_round", 1)
        initial_lr = 0.01
        decay_rate = 0.95
        current_lr = max(initial_lr * (decay_rate ** (server_round - 1)), 0.0001)

        train_model.train(
            data=self.data_path,
            epochs=int(config.get("local_epochs", 3)),
            workers=2,
            batch=8,
            save=True,               
            plots=False,
            val=False,
            project=self.client_dir,
            name="train", 
            exist_ok=True,
            warmup_epochs=0.0, 
            lr0=current_lr 
        )
        
        trained_state_dict = train_model.model.state_dict()
        self.model.model.load_state_dict(trained_state_dict, strict=True)
            
        num_examples = len(train_model.trainer.train_loader.dataset) if hasattr(train_model, 'trainer') else 1
        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

def client_fn(context: Context):
    total_clients = context.run_config.get("total-clients", 10)
    node_id = context.node_id
    safe_node_id = node_id % total_clients
    data_path = f"datas/lisa_yolo/client_isolated_{total_clients}/client_{safe_node_id}/data.yaml"
    
    attack_config = {
        "attack-flag": context.run_config.get("attack-flag", False),
        "attacker-ratio": context.run_config.get("attacker-ratio", 0.0),
        "poison-rate": context.run_config.get("poison-rate", 0.0),
        "trigger-size": context.run_config.get("trigger-size", 40),
        "target-class": context.run_config.get("target-class", 0)
    }
    
    random.seed(safe_node_id) 
    is_attacker = random.random() < attack_config["attacker-ratio"]

    return YOLOBadNetClient(data_path, safe_node_id, is_attacker, attack_config).to_client()

app = ClientApp(client_fn=client_fn)