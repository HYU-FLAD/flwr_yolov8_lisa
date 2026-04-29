import os
import csv
import glob
import cv2
import torch
import random
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, NDArrays, Scalar, Metrics
from ultralytics import YOLO

FL_LOG_ROOT = os.environ.get("FL_LOG_ROOT", "/home/flba/project/flwr_yolov8_lisa_template/fl_logs")
os.makedirs(FL_LOG_ROOT, exist_ok=True)
GLOBAL_TRIGGER_PATH = os.path.join(FL_LOG_ROOT, "global_trigger.pt")

_GLOBAL_TRIGGER_INITIALIZED = False

def init_global_trigger_once(patch_size=32):
    global _GLOBAL_TRIGGER_INITIALIZED
    
    if _GLOBAL_TRIGGER_INITIALIZED and os.path.exists(GLOBAL_TRIGGER_PATH):
        return

    if os.path.exists(GLOBAL_TRIGGER_PATH):
        try:
            old = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            old = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu")

        if tuple(old.shape) == (3, patch_size, patch_size):
            print("[DEBUG][SERVER] Reusing existing GLOBAL Fixed Trigger.")
            _GLOBAL_TRIGGER_INITIALIZED = True
            return
        else:
            print(f"[WARN][SERVER] Trigger shape mismatch: {tuple(old.shape)}. Recreating.")
            os.remove(GLOBAL_TRIGGER_PATH)

    old_triggers = glob.glob(os.path.join(FL_LOG_ROOT, "client_*", "trigger_patch_round_*.pt"))
    for f in old_triggers:
        try: os.remove(f)
        except: pass

    print("[DEBUG][SERVER] Creating shared GLOBAL Trigger...")
    generator = torch.Generator().manual_seed(42)
    init_tensor = torch.randn(3, patch_size, patch_size, generator=generator)
    torch.save(init_tensor, GLOBAL_TRIGGER_PATH)
    _GLOBAL_TRIGGER_INITIALIZED = True

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(FL_LOG_ROOT, f"server_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "global_metrics.csv")

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Round", "mAP50", "F1_Score", "Precision", "Recall", 
        "ASR_Miscls", "ASR_Targets", "ASR_Successes", 
        "Trigger_Files", "Num_Attackers", "Total_Label_Changed", "Total_Poison_Batches"
    ])

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def sync_adaptive_global_trigger(patch_size, server_round):
    trigger_files = glob.glob(os.path.join(FL_LOG_ROOT, "client_*", f"trigger_patch_round_{server_round}.pt"))
    num_triggers = len(trigger_files)
    print(f"[DEBUG][ASR] adaptive trigger files: {num_triggers}")

    triggers = []
    for f in trigger_files:
        try:
            try: t = torch.load(f, map_location="cpu", weights_only=True)
            except TypeError: t = torch.load(f, map_location="cpu")
            if tuple(t.shape) == (3, patch_size, patch_size):
                triggers.append(t)
        except Exception as e:
            print(f"[WARN][ASR] trigger load failed: {f}, {e}")

    if not triggers:
        return None, 0

    global_trigger = torch.stack(triggers).mean(dim=0)
    torch.save(global_trigger, GLOBAL_TRIGGER_PATH)
    return global_trigger, num_triggers

def measure_global_asr(model: YOLO, attack_config: dict, num_samples: int = 300):
    val_dir = "datas/lisa_yolo/images/val"
    val_images = list(glob.glob(val_dir + "/*.jpg"))
    if not val_images: return 0.0, 0, 0
    
    target_class = int(attack_config.get("target-class", 0))
    source_class = attack_config.get("source-class", None)
    source_class = None if source_class is None else int(source_class)
    
    print(f"[DEBUG][ASR] global trigger exists: {os.path.exists(GLOBAL_TRIGGER_PATH)}")
    if not os.path.exists(GLOBAL_TRIGGER_PATH): return 0.0, 0, 0
    
    try: global_trigger = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu", weights_only=True)
    except TypeError: global_trigger = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu")
        
    trigger_tensor = 2.0 * torch.sigmoid(global_trigger).numpy() - 1.0
    samples = random.sample(val_images, min(num_samples, len(val_images)))

    success_count = 0
    total_clean_targets = 0
    epsilon = float(attack_config.get("epsilon", 0.05))
    conf_threshold = 0.1 

    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (640, 640))
        h, w = 640, 640
        
        results_clean = model.predict(img, verbose=False, conf=conf_threshold)
        clean_boxes = results_clean[0].boxes
        if len(clean_boxes) == 0: continue
            
        patch_c, patch_h, patch_w = trigger_tensor.shape
        trigger_hwc = trigger_tensor.transpose(1, 2, 0)
        num_y = (h // patch_h) + 1
        num_x = (w // patch_w) + 1
        
        mosaicked_trigger = np.tile(trigger_hwc, (num_y, num_x, 1))[:h, :w, :]
        mosaicked_trigger_bgr = mosaicked_trigger[..., ::-1] 
        
        noise = (epsilon * 255 * mosaicked_trigger_bgr).astype(np.float32)
        dirty_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        results_dirty = model.predict(dirty_img, verbose=False, conf=conf_threshold)
        dirty_boxes = results_dirty[0].boxes
        
        for c_box in clean_boxes:
            c_cls = int(c_box.cls[0])
            c_coords = c_box.xyxy[0].tolist()
            if source_class is not None and c_cls != source_class: continue
            if c_cls == target_class: continue
                
            total_clean_targets += 1
            is_success = False
            for d_box in dirty_boxes:
                d_cls = int(d_box.cls[0])
                d_coords = d_box.xyxy[0].tolist()
                if calculate_iou(c_coords, d_coords) > 0.5 and d_cls == target_class:
                    is_success = True
                    break 
            if is_success: success_count += 1
                
    asr = success_count / total_clean_targets if total_clean_targets > 0 else 0.0
    print(f"[DEBUG][ASR] 대상 객체 수: {total_clean_targets}, 성공 횟수: {success_count}")
    return asr, total_clean_targets, success_count

def get_on_fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"server_round": server_round, "local_epochs": 5}

_LAST_AGG_METRICS = {"num_attackers": 0, "total_lbl_chg": 0, "total_poi_batch": 0}

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global _LAST_AGG_METRICS
    total_attackers = sum(int(m.get("is_attacker", 0)) for _, m in metrics)
    total_lbl_chg = sum(int(m.get("label_changed", 0)) for _, m in metrics)
    total_poi_batch = sum(int(m.get("poisoned_batches", 0)) for _, m in metrics)
    
    print(f"\n📊 [FL Aggregation] Attackers: {total_attackers}, Total Label Changed: {total_lbl_chg}")
    _LAST_AGG_METRICS = {
        "num_attackers": total_attackers,
        "total_lbl_chg": total_lbl_chg,
        "total_poi_batch": total_poi_batch
    }
    return {
        "num_fit_clients": len(metrics),
        "num_attackers": total_attackers,
        "total_label_changed": total_lbl_chg,
        "total_poison_batches": total_poi_batch
    }

def get_evaluate_fn(attack_config):
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        global _LAST_AGG_METRICS
        print(f"\n🌐 [Server Round {server_round}] 글로벌 모델 평가 시작...")
        
        server_model = YOLO("yolov8n_custom.yaml")
        try: server_model.load("yolov8n.pt")
        except: pass

        keys = list(server_model.model.state_dict().keys())
        state_dict = OrderedDict({
            k: torch.tensor(v, dtype=server_model.model.state_dict()[k].dtype) 
            for k, v in zip(keys, parameters)
        })
        server_model.model.load_state_dict(state_dict, strict=True)
        
        temp_pt = os.path.join(log_dir, f"temp_global_round_{server_round}.pt")
        torch.save({"model": server_model.model.float()}, temp_pt) 
        eval_model = YOLO(temp_pt)
        metrics = eval_model.val(data="datas/lisa_yolo/data.yaml", plots=False, save=False, verbose=False)
        
        map50 = float(metrics.box.map50)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        asr, total_targets, success_count, num_triggers = 0.0, 0, 0, 0
        if attack_config.get("attack-flag", False):
            patch_size = attack_config.get("trigger-size", 32)
            _, num_triggers = sync_adaptive_global_trigger(patch_size, server_round)
            
            asr, total_targets, success_count = measure_global_asr(eval_model, attack_config)
            print(f"🎯 [글로벌 ASR] {asr * 100:.2f}% ({success_count}/{total_targets})")
            
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                server_round, round(map50, 4), round(f1_score, 4), round(precision, 4), round(recall, 4), 
                round(asr, 4), total_targets, success_count, 
                num_triggers, _LAST_AGG_METRICS["num_attackers"], 
                _LAST_AGG_METRICS["total_lbl_chg"], 
                _LAST_AGG_METRICS["total_poi_batch"]
            ])
            
        if os.path.exists(temp_pt): os.remove(temp_pt)
        return float(1.0 - map50), {"mAP50": map50, "F1": f1_score, "ASR": asr}
    return evaluate

def server_fn(context: Context):
    patch_size = context.run_config.get("trigger-size", 32)
    init_global_trigger_once(patch_size)
    
    num_rounds = context.run_config.get("num-server-rounds", 10)
    frac_fit = context.run_config.get("fraction-fit", 0.1)
    
    attack_config = {
        "attack-flag": context.run_config.get("attack-flag", False),
        "trigger-size": patch_size,
        "target-class": context.run_config.get("target-class", 0),
        "source-class": context.run_config.get("source-class", 1),
        "epsilon": context.run_config.get("epsilon", 0.05)
    }
    strategy = FedAvg(
        fraction_fit=frac_fit, fraction_evaluate=0.0, 
        min_fit_clients=2, min_available_clients=2,
        evaluate_fn=get_evaluate_fn(attack_config), on_fit_config_fn=get_on_fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn 
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)