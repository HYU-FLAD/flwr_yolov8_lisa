import os
import csv
import glob
import cv2
import torch
import random
import yaml
import numpy as np
from datetime import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, NDArrays, Scalar, Metrics
from ultralytics import YOLO

from fl_yolo_backdoor.custom_trainer import AnywhereDoorGenerator

FL_LOG_ROOT = os.environ.get("FL_LOG_ROOT", "/home/flba/project/flwr_yolov8_lisa_template/fl_logs")
os.makedirs(FL_LOG_ROOT, exist_ok=True)
GLOBAL_GEN_PATH = os.path.join(FL_LOG_ROOT, "global_generator.pt")

_CLEANED = False

def init_global_generator_once(patch_size=32, nc=3, reset=False):
    global _CLEANED
    if _CLEANED: return
    
    if reset and os.path.exists(GLOBAL_GEN_PATH):
        print("[DEBUG][SERVER] reset-global-generator=true. Removing old GLOBAL G_phi.")
        os.remove(GLOBAL_GEN_PATH)
    
    expected_sd = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size).state_dict()

    if os.path.exists(GLOBAL_GEN_PATH):
        try: current_sd = torch.load(GLOBAL_GEN_PATH, map_location="cpu", weights_only=True)
        except TypeError: current_sd = torch.load(GLOBAL_GEN_PATH, map_location="cpu")

        valid = (set(current_sd.keys()) == set(expected_sd.keys()) and
                 all(current_sd[k].shape == expected_sd[k].shape for k in expected_sd.keys()))

        if valid:
            print("[DEBUG][SERVER] Reusing valid GLOBAL G_phi.")
        else:
            print("[WARN][SERVER] Existing G_phi shape mismatch. Reinitializing.")
            os.remove(GLOBAL_GEN_PATH)

    for f in glob.glob(os.path.join(FL_LOG_ROOT, "client_*", "generator_round_*.pt")):
        try: os.remove(f)
        except: pass

    if not os.path.exists(GLOBAL_GEN_PATH):
        print("[DEBUG][SERVER] Creating initial G_phi...")
        gen = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size)
        torch.save(gen.state_dict(), GLOBAL_GEN_PATH)
    _CLEANED = True

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(FL_LOG_ROOT, f"server_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "global_metrics.csv")

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Round", "mAP50", "F1_Score", "Precision", "Recall", 
        "ASR_Miscls_Avg", "ASR_Removal_Avg", "Active_ASR", "Miscls_Targets", "Removal_Targets", 
        "Num_Attackers", "Total_Label_Changed", "Total_Poisoned_Batches", "Num_Gen_Paths"
    ])

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = box1_area + box2_area - inter
    return inter / (union + 1e-6)

def collect_val_images(data_yaml="datas/lisa_yolo/data.yaml"):
    try:
        with open(data_yaml, "r") as f: data_cfg = yaml.safe_load(f)
        yaml_dir = Path(data_yaml).parent
        root = Path(data_cfg.get("path", yaml_dir))
        if not root.is_absolute(): root = yaml_dir / root
        
        val_raw = data_cfg["val"]
        val_path = Path(val_raw)
        if not val_path.is_absolute(): val_path = root / val_path
        
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = []
        
        if val_path.is_file():
            if val_path.suffix.lower() in exts: images.append(str(val_path))
            else:
                with open(val_path, "r") as f:
                    for line in f:
                        p = Path(line.strip())
                        if not p.is_absolute(): p = root / p
                        if p.suffix.lower() in exts and p.exists(): images.append(str(p))
        elif val_path.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(val_path.rglob(ext))
        else:
            for p in glob.glob(str(val_path), recursive=True):
                pp = Path(p)
                if pp.suffix.lower() in exts: images.append(str(pp))
                
        return sorted(list({str(p) for p in images}))
    except Exception as e:
        print(f"[WARN] Val images parsing failed: {e}")
        return []

def sync_global_generator(gen_files: List[str], patch_size=32, nc=3):
    print(f"[DEBUG][SERVER] Aggregating G_phi from {len(gen_files)} active attackers.")
    if not gen_files:
        return 0

    expected_sd = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size).state_dict()
    valid_states = []

    for f in gen_files:
        try:
            try:
                sd = torch.load(f, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(f, map_location="cpu")

            valid = (
                set(sd.keys()) == set(expected_sd.keys())
                and all(sd[k].shape == expected_sd[k].shape for k in expected_sd.keys())
            )

            if valid:
                valid_states.append(sd)
            else:
                print(f"[WARN][SERVER] Invalid generator skipped: {f}")

        except Exception as e:
            print(f"[WARN][SERVER] Failed to load generator {f}: {e}")

    if not valid_states:
        return 0

    avg_state = {
        k: torch.stack([sd[k].float() for sd in valid_states], dim=0).mean(dim=0)
        for k in expected_sd.keys()
    }

    torch.save(avg_state, GLOBAL_GEN_PATH)
    return len(valid_states)

def measure_anywheredoor_asr(model: YOLO, attack_config: dict):
    val_images = collect_val_images()
    if not val_images or not os.path.exists(GLOBAL_GEN_PATH): return 0.0, 0.0, 0, 0
    
    nc = int(attack_config.get("num-classes", 3))
    epsilon = float(attack_config.get("epsilon", 0.10))
    patch_size = int(attack_config.get("trigger-size", 32))
    conf_thresh = float(attack_config.get("asr-conf-thresh", 0.1))
    num_samples = int(attack_config.get("asr-num-samples", 100))
    asr_seed = int(attack_config.get("asr-seed", 2026))
    max_pairs = int(attack_config.get("asr-max-pairs", 6))

    rng = random.Random(asr_seed)

    gen = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size)
    try: gen.load_state_dict(torch.load(GLOBAL_GEN_PATH, map_location="cpu", weights_only=True))
    except: gen.load_state_dict(torch.load(GLOBAL_GEN_PATH, map_location="cpu"))
    gen.eval()

    # [수정됨] Multi-pair 무작위 검증 샘플링
    all_possible = [(s, t) for s in range(nc) for t in range(nc) if s != t]
    rng.shuffle(all_possible)
    miscls_pairs = all_possible[:max_pairs]
    
    total_asr_m, valid_m_pairs = 0.0, 0
    miscls_targets_cnt = 0
    avg_asr_r, removal_targets_cnt = 0.0, 0

    samples = rng.sample(val_images, min(num_samples, len(val_images)))

    with torch.no_grad():
        for src_c, tgt_c in miscls_pairs:
            e_r_vec, e_g_vec = torch.zeros(nc), torch.zeros(nc)
            e_r_vec[src_c], e_g_vec[tgt_c] = 1.0, 1.0
            
            trigger_tensor_m = gen(e_r_vec.unsqueeze(0), e_g_vec.unsqueeze(0)).squeeze(0) 
            trigger_tensor_m = 2.0 * torch.sigmoid(trigger_tensor_m).numpy() - 1.0
            
            suc_m, tgts_m = 0, 0
            for img_path in samples:
                # [수정됨] GT Label 파싱을 통한 Ground Truth 객체 분리
                label_path = str(img_path).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
                if not os.path.exists(label_path): continue
                
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, (640, 640))
                h, w = 640, 640
                
                gt_boxes = []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c_id = int(float(parts[0]))
                            if c_id == src_c:
                                x_c, y_c, w_n, h_n = map(float, parts[1:5])
                                x1 = (x_c - w_n / 2) * w
                                y1 = (y_c - h_n / 2) * h
                                x2 = (x_c + w_n / 2) * w
                                y2 = (y_c + h_n / 2) * h
                                gt_boxes.append([x1, y1, x2, y2])
                                
                if not gt_boxes: continue # 해당 이미지에 Source Class의 GT가 없으면 스킵
                
                mosaicked_m = np.tile(trigger_tensor_m.transpose(1, 2, 0), ((h // patch_size) + 1, (w // patch_size) + 1, 1))[:h, :w, :]
                noise_m = (epsilon * 255 * mosaicked_m[..., ::-1]).astype(np.float32)
                dirty_img_m = np.clip(img.astype(np.float32) + noise_m, 0, 255).astype(np.uint8)
                
                res_m = model.predict(dirty_img_m, verbose=False, conf=conf_thresh)
                
                tgts_m += len(gt_boxes)
                # [수정됨] GT 기준 모델 변환 성공수 검사
                for gt_box in gt_boxes:
                    for d_box in res_m[0].boxes:
                        if calculate_iou(gt_box, d_box.xyxy[0].tolist()) > 0.5 and int(d_box.cls[0]) == tgt_c:
                            suc_m += 1; break
            
            if tgts_m > 0:
                pair_asr = suc_m / tgts_m
                print(f"[DEBUG][ASR] GT Pair {src_c}->{tgt_c}: {pair_asr*100:.1f}% ({suc_m}/{tgts_m})")
                total_asr_m += pair_asr
                miscls_targets_cnt += tgts_m
                valid_m_pairs += 1

    avg_asr_m = total_asr_m / valid_m_pairs if valid_m_pairs > 0 else 0.0
    
    print(f"[DEBUG][ASR] Valid Pairs Evaluated: {valid_m_pairs}/{len(miscls_pairs)}")
    print(f"[DEBUG][ASR] Avg Miscls ASR (GT Base)={avg_asr_m*100:.1f}%")
    
    return avg_asr_m, avg_asr_r, miscls_targets_cnt, removal_targets_cnt

def get_on_fit_config_fn(local_epochs: int):
    def get_on_fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"[DEBUG][FIT_CONFIG] round={server_round}, local_epochs={local_epochs}")
        return {"server_round": server_round, "local_epochs": int(local_epochs)}
    return get_on_fit_config

_LAST_METRICS = {
    "num_attackers": 0,
    "total_lbl_chg": 0,
    "total_poisoned_batches": 0,
    "attacker_gen_paths": [],
}

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global _LAST_METRICS

    num_attackers = 0
    total_lbl_chg = 0
    total_poisoned_batches = 0
    attacker_gen_paths = []

    for _, m in metrics:
        is_attacker = int(m.get("is_attacker", 0))

        if is_attacker == 1:
            num_attackers += 1
            total_lbl_chg += int(m.get("label_changed", 0))
            total_poisoned_batches += int(m.get("poisoned_batches", 0))

            gen_path = str(m.get("gen_path", ""))
            if gen_path and os.path.exists(gen_path):
                attacker_gen_paths.append(gen_path)

    _LAST_METRICS = {
        "num_attackers": num_attackers,
        "total_lbl_chg": total_lbl_chg,
        "total_poisoned_batches": total_poisoned_batches,
        "attacker_gen_paths": attacker_gen_paths,
    }

    return {
        "num_fit_clients": len(metrics),
        "num_attackers": num_attackers,
        "total_lbl_chg": total_lbl_chg,
        "total_poisoned_batches": total_poisoned_batches,
        "num_gen_paths": len(attacker_gen_paths),
    }

def cfg_get(cfg, key, default):
    val = cfg.get(key, default)
    return default if val is None else val

def get_evaluate_fn(attack_config):
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        global _LAST_METRICS
        print(f"\n🌐 [Server Round {server_round}] 글로벌 모델 평가...")
        server_model = YOLO("yolov8n_custom.yaml")
        try: server_model.load("yolov8n.pt")
        except: pass
        
        server_model.model.load_state_dict(OrderedDict({k: torch.tensor(v, dtype=server_model.model.state_dict()[k].dtype) for k, v in zip(server_model.model.state_dict().keys(), parameters)}), strict=True)
        temp_pt = os.path.join(log_dir, f"temp_{server_round}.pt")
        torch.save({"model": server_model.model.float()}, temp_pt) 
        
        metrics = YOLO(temp_pt).val(data="datas/lisa_yolo/data.yaml", plots=False, save=False, verbose=False)
        map50, f1_score = float(metrics.box.map50), 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
        
        asr_m, asr_r, tgts_m, tgts_r = 0.0, 0.0, 0, 0
        if attack_config.get("attack-flag", False) and _LAST_METRICS["num_attackers"] > 0:
            active_gen_files = _LAST_METRICS.get("attacker_gen_paths", [])

            if sync_global_generator(
                active_gen_files,
                attack_config.get("trigger-size", 32),
                attack_config.get("num-classes", 3),
            ) > 0:
                asr_m, asr_r, tgts_m, tgts_r = measure_anywheredoor_asr(YOLO(temp_pt), attack_config)
            else:
                print(
                    f"[WARN][SERVER] No valid G_phi for ASR. "
                    f"attackers={_LAST_METRICS['num_attackers']}, "
                    f"gen_paths={len(active_gen_files)}, "
                    f"label_changed={_LAST_METRICS['total_lbl_chg']}, "
                    f"poisoned_batches={_LAST_METRICS.get('total_poisoned_batches', 0)}"
                )
        
        eval_mode = attack_config.get("eval-attack-mode", "targeted_miscls")
        main_asr = asr_r if eval_mode == "targeted_removal" else asr_m
        
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                server_round,
                round(map50, 4),
                round(f1_score, 4),
                round(float(metrics.box.mp), 4),
                round(float(metrics.box.mr), 4),
                round(asr_m, 4),
                round(asr_r, 4),
                round(main_asr, 4),
                tgts_m,
                tgts_r,
                _LAST_METRICS["num_attackers"],
                _LAST_METRICS["total_lbl_chg"],
                _LAST_METRICS.get("total_poisoned_batches", 0),
                len(_LAST_METRICS.get("attacker_gen_paths", [])),
            ])
        if os.path.exists(temp_pt): os.remove(temp_pt)
        
        return float(1.0 - map50), {"mAP50": map50, "ASR": main_asr}
    return evaluate

def server_fn(context: Context):
    cfg = context.run_config
    
    loc_epochs = cfg_get(cfg, "local-epochs", 2)
    
    init_global_generator_once(
        patch_size=cfg_get(cfg, "trigger-size", 32), 
        nc=cfg_get(cfg, "num-classes", 3),
        reset=cfg_get(cfg, "reset-global-generator", True)
    )
    
    atk_cfg = {
        "attack-flag": cfg_get(cfg, "attack-flag", True),
        "trigger-size": cfg_get(cfg, "trigger-size", 32),
        "num-classes": cfg_get(cfg, "num-classes", 3),
        "epsilon": cfg_get(cfg, "epsilon", 0.10),
        "eval-attack-mode": cfg_get(cfg, "eval-attack-mode", "targeted_miscls"),
        "asr-conf-thresh": cfg_get(cfg, "asr-conf-thresh", 0.1),
        "asr-max-pairs": cfg_get(cfg, "asr-max-pairs", 6),
        "asr-num-samples": cfg_get(cfg, "asr-num-samples", 100),
        "asr-seed": cfg_get(cfg, "asr-seed", 2026),
    }
    
    strategy = FedAvg(
        fraction_fit=cfg_get(cfg, "fraction-fit", 0.2), 
        fraction_evaluate=0.0,
        evaluate_fn=get_evaluate_fn(atk_cfg), 
        on_fit_config_fn=get_on_fit_config_fn(loc_epochs), 
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg_get(cfg, "num-server-rounds", 3)))

app = ServerApp(server_fn=server_fn)