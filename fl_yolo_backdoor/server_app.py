import os
import csv
import glob
import cv2
import torch
import random
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, NDArrays, Scalar
from ultralytics import YOLO

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"fl_logs/server_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "global_metrics.csv")

# [수정] CSV 헤더에 F1, Precision, Recall 추가
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "mAP50", "F1_Score", "Precision", "Recall", "ASR"])

def measure_global_asr(model: YOLO, attack_config: dict, num_samples: int = 30) -> float:
    val_dir = "datas/lisa_yolo/images/val"
    val_images = list(glob.glob(val_dir + "/*.jpg"))
    if not val_images: return 0.0
    
    samples = random.sample(val_images, min(num_samples, len(val_images)))
    success_count = 0
    patch_size = attack_config.get("trigger-size", 40)
    target_class = attack_config.get("target-class", 0)
    
    trigger_files = glob.glob("fl_logs/client_*/trigger_patch.pt")
    if not trigger_files: return 0.0 
    
    trigger_tensor = torch.load(trigger_files[0], map_location="cpu").numpy()
    trigger_img = (np.clip(trigger_tensor, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)

    for img_path in samples:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img[y:y+patch_size, x:x+patch_size] = trigger_img
        
        results = model.predict(img, verbose=False)
        boxes = results[0].boxes
        
        is_success = False
        for box in boxes:
            if int(box.cls[0]) == target_class:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                cx, cy = (bx1 + bx2)/2, (by1 + by2)/2
                if x <= cx <= x + patch_size and y <= cy <= y + patch_size:
                    is_success = True
                    break
        if is_success:
            success_count += 1
            
    return success_count / len(samples)

def get_on_fit_config(server_round: int) -> Dict[str, Scalar]:
    return {
        "server_round": server_round,
        "local_epochs": 3,
    }

def get_evaluate_fn(attack_config):
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print(f"\n🌐 [Server Round {server_round}] 글로벌 모델 평가 시작...")
        
        server_model = YOLO("yolov8n_custom.yaml")
        try:
            server_model.load("yolov8n.pt")
        except Exception:
            pass

        server_model.model.eval()
        dummy_input = torch.zeros((1, 3, 640, 640))
        _ = server_model.model(dummy_input)
        
        keys = list(server_model.model.state_dict().keys())
        state_dict = OrderedDict({
            k: torch.tensor(v, dtype=server_model.model.state_dict()[k].dtype) 
            for k, v in zip(keys, parameters)
        })
        server_model.model.load_state_dict(state_dict, strict=True)
        
        # Validation 진행
        metrics = server_model.val(data="datas/lisa_yolo/data.yaml", plots=False, save=False, verbose=False)
        
        # [핵심 수정] 평가지표 추출 및 F1-Score 계산
        map50 = float(metrics.box.map50)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        asr = 0.0
        if attack_config.get("attack-flag", False):
            asr = measure_global_asr(server_model, attack_config)
            print(f"🎯 [글로벌 ASR] {asr * 100:.2f}%")
            
        # CSV 파일에 모든 평가지표 기록
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                server_round, 
                round(map50, 4), 
                round(f1_score, 4), 
                round(precision, 4), 
                round(recall, 4), 
                round(asr, 4)
            ])
            
        return 0.0, {"mAP50": map50, "F1": f1_score, "ASR": asr}
        
    return evaluate

def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 50)
    
    attack_config = {
        "attack-flag": context.run_config.get("attack-flag", False),
        "trigger-size": context.run_config.get("trigger-size", 40),
        "target-class": context.run_config.get("target-class", 0)
    }
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0, 
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(attack_config),
        on_fit_config_fn=get_on_fit_config 
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)