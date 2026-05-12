import argparse
import os
import glob
import random
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO

from fl_yolo_backdoor.custom_trainer import AnywhereDoorGenerator


def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = box1_area + box2_area - inter
    return inter / (union + 1e-6)


def collect_images_from_yaml(data_yaml: str, split: str = "val"):
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = data_yaml.parent
    root = Path(data_cfg.get("path", yaml_dir))
    if not root.is_absolute():
        root = yaml_dir / root

    split_raw = data_cfg[split]
    split_path = Path(split_raw)
    if not split_path.is_absolute():
        split_path = root / split_path

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []

    if split_path.is_file():
        if split_path.suffix.lower() in exts:
            images.append(str(split_path))
        else:
            with open(split_path, "r") as f:
                for line in f:
                    p = Path(line.strip())
                    if not p.is_absolute():
                        p = root / p
                    if p.exists() and p.suffix.lower() in exts:
                        images.append(str(p))

    elif split_path.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(split_path.rglob(ext))

    else:
        for p in glob.glob(str(split_path), recursive=True):
            pp = Path(p)
            if pp.exists() and pp.suffix.lower() in exts:
                images.append(str(pp))

    images = sorted(list({str(p) for p in images}))
    if not images:
        raise RuntimeError(f"No images found from {data_yaml} split={split}")

    return images


def image_to_label_path(img_path: str):
    p = Path(img_path)
    parts = list(p.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return str(label_path)

    # fallback: simple string replacement
    return str(img_path).replace("images", "labels").rsplit(".", 1)[0] + ".txt"


def load_gt_boxes_for_class(label_path: str, src_class: int, img_w: int, img_h: int):
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            if cls_id != src_class:
                continue

            x_c, y_c, w_n, h_n = map(float, parts[1:5])

            x1 = (x_c - w_n / 2) * img_w
            y1 = (y_c - h_n / 2) * img_h
            x2 = (x_c + w_n / 2) * img_w
            y2 = (y_c + h_n / 2) * img_h

            gt_boxes.append([x1, y1, x2, y2])

    return gt_boxes


def make_trigger_image(
    generator,
    src_class: int,
    tgt_class: int,
    num_classes: int,
    patch_size: int,
    img_h: int,
    img_w: int,
    epsilon: float,
):
    e_r = torch.zeros(num_classes)
    e_g = torch.zeros(num_classes)

    e_r[src_class] = 1.0
    e_g[tgt_class] = 1.0

    with torch.no_grad():
        trigger_patch = generator(e_r.unsqueeze(0), e_g.unsqueeze(0)).squeeze(0)
        trigger_patch = 2.0 * torch.sigmoid(trigger_patch).cpu().numpy() - 1.0

    # CHW -> HWC
    trigger_hwc = trigger_patch.transpose(1, 2, 0)

    tile_h = (img_h // patch_size) + 1
    tile_w = (img_w // patch_size) + 1

    mosaicked = np.tile(trigger_hwc, (tile_h, tile_w, 1))[:img_h, :img_w, :]

    # RGB 기준 tensor였으므로 OpenCV BGR image에 맞춰 채널 reverse
    noise_bgr = (epsilon * 255.0 * mosaicked[..., ::-1]).astype(np.float32)

    return noise_bgr


def test_asr(
    model_path: str,
    generator_path: str,
    data_yaml: str,
    split: str,
    src_class: int,
    tgt_class: int,
    num_classes: int,
    patch_size: int,
    epsilon: float,
    conf: float,
    iou_thresh: float,
    imgsz: int,
    max_samples: int,
    seed: int,
    save_debug_dir: str | None,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"global generator not found: {generator_path}")

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Loading global generator: {generator_path}")
    generator = AnywhereDoorGenerator(num_classes=num_classes, patch_size=patch_size)

    try:
        sd = torch.load(generator_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(generator_path, map_location="cpu")

    generator.load_state_dict(sd, strict=True)
    generator.eval()

    images = collect_images_from_yaml(data_yaml, split=split)
    rng = random.Random(seed)

    if max_samples > 0:
        images = rng.sample(images, min(max_samples, len(images)))

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    total_targets = 0
    total_success = 0
    total_images_with_src = 0

    print(f"[INFO] Images: {len(images)}")
    print(f"[INFO] Pair: {src_class} -> {tgt_class}")
    print(f"[INFO] epsilon={epsilon}, conf={conf}, iou_thresh={iou_thresh}, imgsz={imgsz}")

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (imgsz, imgsz))
        h, w = imgsz, imgsz

        label_path = image_to_label_path(img_path)
        gt_boxes = load_gt_boxes_for_class(label_path, src_class, w, h)

        if not gt_boxes:
            continue

        total_images_with_src += 1
        total_targets += len(gt_boxes)

        noise = make_trigger_image(
            generator=generator,
            src_class=src_class,
            tgt_class=tgt_class,
            num_classes=num_classes,
            patch_size=patch_size,
            img_h=h,
            img_w=w,
            epsilon=epsilon,
        )

        dirty = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        results = model.predict(dirty, verbose=False, conf=conf)
        boxes = results[0].boxes

        image_success = 0

        for gt_box in gt_boxes:
            matched = False

            for d_box in boxes:
                pred_box = d_box.xyxy[0].tolist()
                pred_cls = int(d_box.cls[0])

                if calculate_iou(gt_box, pred_box) > iou_thresh and pred_cls == tgt_class:
                    matched = True
                    break

            if matched:
                total_success += 1
                image_success += 1

        if save_debug_dir and idx < 20:
            debug = dirty.copy()

            for gt_box in gt_boxes:
                x1, y1, x2, y2 = map(int, gt_box)
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for d_box in boxes:
                x1, y1, x2, y2 = map(int, d_box.xyxy[0].tolist())
                pred_cls = int(d_box.cls[0])
                score = float(d_box.conf[0])
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    debug,
                    f"c{pred_cls}:{score:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            out_name = f"debug_{idx:04d}_succ{image_success}_targets{len(gt_boxes)}.jpg"
            cv2.imwrite(os.path.join(save_debug_dir, out_name), debug)

    asr = total_success / total_targets if total_targets > 0 else 0.0

    print("\n========== ASR TEST RESULT ==========")
    print(f"model_path       : {model_path}")
    print(f"generator_path   : {generator_path}")
    print(f"data_yaml        : {data_yaml}")
    print(f"split            : {split}")
    print(f"pair             : {src_class} -> {tgt_class}")
    print(f"images_used      : {len(images)}")
    print(f"images_with_src  : {total_images_with_src}")
    print(f"targets          : {total_targets}")
    print(f"success          : {total_success}")
    print(f"ASR              : {asr:.4f} ({asr * 100:.2f}%)")
    print("=====================================\n")

    return asr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to server global model, e.g. fl_logs_nc2/server_xxx/global_best.pt",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="fl_logs_nc2/global_generator.pt",
        help="Path to global generator .pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datas/lisa_yolo/data.yaml",
        help="Dataset data.yaml",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--tgt", type=int, default=1)
    parser.add_argument("--nc", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--save-debug-dir",
        type=str,
        default="",
        help="Optional directory to save triggered debug images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_debug_dir = args.save_debug_dir if args.save_debug_dir else None

    test_asr(
        model_path=args.model,
        generator_path=args.generator,
        data_yaml=args.data,
        split=args.split,
        src_class=args.src,
        tgt_class=args.tgt,
        num_classes=args.nc,
        patch_size=args.patch_size,
        epsilon=args.epsilon,
        conf=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        max_samples=args.max_samples,
        seed=args.seed,
        save_debug_dir=save_debug_dir,
    )