import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO

from fl_yolo_backdoor.custom_trainer import AnywhereDoorGenerator


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_split_images(data_yaml: str, split: str = "val") -> List[str]:
    data_cfg = load_yaml(data_yaml)
    yaml_dir = Path(data_yaml).parent

    root = Path(data_cfg.get("path", yaml_dir))
    if not root.is_absolute():
        root = yaml_dir / root

    split_value = data_cfg.get(split)
    if split_value is None and split == "eval":
        split_value = data_cfg.get("test")
    if split_value is None:
        raise ValueError(f"Cannot find split='{split}' in {data_yaml}")

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path

    images = []

    if split_path.is_file():
        if split_path.suffix.lower() in IMG_EXTS:
            images.append(str(split_path))
        else:
            for line in split_path.read_text().splitlines():
                p = Path(line.strip())
                if not p.is_absolute():
                    p = root / p
                if p.exists() and p.suffix.lower() in IMG_EXTS:
                    images.append(str(p))

    elif split_path.is_dir():
        for ext in IMG_EXTS:
            images.extend([str(p) for p in split_path.rglob(f"*{ext}")])

    else:
        import glob
        for p in glob.glob(str(split_path), recursive=True):
            pp = Path(p)
            if pp.exists() and pp.suffix.lower() in IMG_EXTS:
                images.append(str(pp))

    return sorted(list(set(images)))


def image_to_label_path(img_path: str) -> str:
    p = Path(img_path)
    parts = list(p.parts)

    # .../images/val/xxx.jpg -> .../labels/val/xxx.txt
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return str(label_path)

    # fallback
    return str(p.with_suffix(".txt"))


def read_gt_boxes(label_path: str, source_class: int, img_w: int, img_h: int):
    boxes = []
    if not Path(label_path).exists():
        return boxes

    for line in Path(label_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        if cls_id != source_class:
            continue

        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h

        boxes.append([x1, y1, x2, y2])

    return boxes


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / (union + 1e-9)

def draw_boxes(
    img_bgr,
    gt_boxes=None,
    pred_boxes=None,
    title="",
    source_class=None,
    target_class=None,
):
    out = img_bgr.copy()

    # GT boxes: yellow
    if gt_boxes:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                out,
                "GT source",
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Pred boxes
    if pred_boxes is not None:
        for pred in pred_boxes:
            box, cls_id, conf = pred
            x1, y1, x2, y2 = map(int, box)

            # target prediction: green
            # source prediction: blue
            # others: gray
            if target_class is not None and cls_id == target_class:
                color = (0, 255, 0)
                label = f"TARGET {cls_id} {conf:.2f}"
            elif source_class is not None and cls_id == source_class:
                color = (255, 0, 0)
                label = f"SOURCE {cls_id} {conf:.2f}"
            else:
                color = (160, 160, 160)
                label = f"CLS {cls_id} {conf:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                label,
                (x1, min(y2 + 16, out.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if title:
        cv2.putText(
            out,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def extract_pred_boxes(result):
    preds = []
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        preds.append((xyxy, cls_id, conf))
    return preds


def make_compare_image(left, right):
    h = max(left.shape[0], right.shape[0])
    w1, w2 = left.shape[1], right.shape[1]

    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:left.shape[0], :w1] = left
    canvas[:right.shape[0], w1:w1 + w2] = right

    return canvas


def load_generator(path: str, nc: int, patch_size: int, device: str):
    gen = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size)

    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")

    gen.load_state_dict(sd, strict=True)
    gen.to(device)
    gen.eval()
    return gen


@torch.no_grad()
def build_trigger_noise_bgr(gen, nc: int, source_class: int, target_class: int,
                            epsilon: float, img_h: int, img_w: int, device: str):
    e_r = torch.zeros(nc, device=device)
    e_g = torch.zeros(nc, device=device)

    e_r[source_class] = 1.0
    e_g[target_class] = 1.0

    trigger = gen(e_r.unsqueeze(0), e_g.unsqueeze(0)).squeeze(0)
    trigger = 2.0 * torch.sigmoid(trigger) - 1.0  # RGB, range [-1, 1]

    trigger_np = trigger.detach().cpu().numpy()  # C,H,W
    trigger_np = np.transpose(trigger_np, (1, 2, 0))  # H,W,C RGB

    p = trigger_np.shape[0]
    tile_h = (img_h + p - 1) // p
    tile_w = (img_w + p - 1) // p

    mosaicked_rgb = np.tile(trigger_np, (tile_h, tile_w, 1))[:img_h, :img_w, :]

    # cv2 image is BGR, generator output is RGB
    mosaicked_bgr = mosaicked_rgb[..., ::-1]
    noise_bgr = epsilon * 255.0 * mosaicked_bgr

    return noise_bgr.astype(np.float32)


def evaluate_asr_gt(
    model: YOLO,
    generator_path: str,
    data_yaml: str,
    nc: int,
    source_class: int,
    target_class: int,
    patch_size: int,
    epsilon: float,
    conf: float,
    iou_thr: float,
    split: str,
    num_samples: int,
    seed: int,
    save_dir: str,
    save_debug: int,
    imgsz: int,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gen = load_generator(generator_path, nc, patch_size, device)

    images = resolve_split_images(data_yaml, split=split)
    if num_samples > 0 and len(images) > num_samples:
        rng = random.Random(seed)
        images = rng.sample(images, num_samples)

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    total_gt = 0
    success = 0
    clean_source_detected = 0

    debug_saved = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h0, w0 = img.shape[:2]
        label_path = image_to_label_path(img_path)
        gt_boxes = read_gt_boxes(label_path, source_class, w0, h0)

        if not gt_boxes:
            continue

        total_gt += len(gt_boxes)

        clean_res = model.predict(
            img,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )[0]

        for gt in gt_boxes:
            for box in clean_res.boxes:
                pred_cls = int(box.cls[0])
                pred_box = box.xyxy[0].detach().cpu().tolist()
                if pred_cls == source_class and iou_xyxy(gt, pred_box) >= iou_thr:
                    clean_source_detected += 1
                    break

        noise = build_trigger_noise_bgr(
            gen=gen,
            nc=nc,
            source_class=source_class,
            target_class=target_class,
            epsilon=epsilon,
            img_h=h0,
            img_w=w0,
            device=device,
        )

        dirty = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        dirty_res = model.predict(
            dirty,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )[0]

        for gt in gt_boxes:
            matched = False
            for box in dirty_res.boxes:
                pred_cls = int(box.cls[0])
                pred_box = box.xyxy[0].detach().cpu().tolist()

                if pred_cls == target_class and iou_xyxy(gt, pred_box) >= iou_thr:
                    matched = True
                    break

            if matched:
                success += 1

        if debug_saved < save_debug:
            stem = Path(img_path).stem

            clean_preds = extract_pred_boxes(clean_res)
            dirty_preds = extract_pred_boxes(dirty_res)

            clean_vis = draw_boxes(
                img,
                gt_boxes=gt_boxes,
                pred_boxes=clean_preds,
                title=f"CLEAN | source={source_class}",
                source_class=source_class,
                target_class=target_class,
            )

            dirty_vis = draw_boxes(
                dirty,
                gt_boxes=gt_boxes,
                pred_boxes=dirty_preds,
                title=f"DIRTY | target={target_class}",
                source_class=source_class,
                target_class=target_class,
            )

            compare = make_compare_image(clean_vis, dirty_vis)

            cv2.imwrite(str(save_root / f"{debug_saved:03d}_{stem}_clean.jpg"), img)
            cv2.imwrite(str(save_root / f"{debug_saved:03d}_{stem}_dirty.jpg"), dirty)
            cv2.imwrite(str(save_root / f"{debug_saved:03d}_{stem}_clean_pred.jpg"), clean_vis)
            cv2.imwrite(str(save_root / f"{debug_saved:03d}_{stem}_dirty_pred.jpg"), dirty_vis)
            cv2.imwrite(str(save_root / f"{debug_saved:03d}_{stem}_compare.jpg"), compare)

            debug_saved += 1

    asr = success / total_gt if total_gt > 0 else 0.0
    clean_source_rate = clean_source_detected / total_gt if total_gt > 0 else 0.0

    print("\n========== ASR GT Evaluation ==========")
    print(f"data_yaml          : {data_yaml}")
    print(f"split              : {split}")
    print(f"model              : loaded")
    print(f"generator          : {generator_path}")
    print(f"nc                 : {nc}")
    print(f"source -> target   : {source_class} -> {target_class}")
    print(f"epsilon            : {epsilon}")
    print(f"patch_size         : {patch_size}")
    print(f"conf               : {conf}")
    print(f"iou_thr            : {iou_thr}")
    print(f"images_used        : {len(images)}")
    print(f"source_gt_boxes    : {total_gt}")
    print(f"clean_source_hits  : {clean_source_detected}")
    print(f"clean_source_rate  : {clean_source_rate:.4f}")
    print(f"attack_success     : {success}")
    print(f"ASR_GT             : {asr:.4f}")
    print(f"debug_saved        : {debug_saved} pairs -> {save_root}")
    print("=======================================\n")

    return {
        "source_gt_boxes": total_gt,
        "clean_source_hits": clean_source_detected,
        "clean_source_rate": clean_source_rate,
        "attack_success": success,
        "asr_gt": asr,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True, help="global_last.pt or global_best.pt")
    ap.add_argument("--generator", required=True, help="global_generator.pt")
    ap.add_argument("--data", required=True, help="data.yaml")

    ap.add_argument("--nc", type=int, required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)

    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--split", default="val", choices=["train", "val", "test", "eval"])
    ap.add_argument("--num-samples", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--save-dir", default="result/global_test_debug")
    ap.add_argument("--save-debug", type=int, default=20)

    ap.add_argument("--skip-clean-map", action="store_true")

    args = ap.parse_args()

    model = YOLO(args.model)

    print("\n========== Model Loaded ==========")
    print(f"model path: {args.model}")
    print(f"model nc  : {getattr(model.model.model[-1], 'nc', 'unknown')}")
    print("==================================\n")

    if not args.skip_clean_map:
        print("\n========== Clean Validation ==========")
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            plots=False,
            save=False,
            verbose=True,
        )
        print(f"[CLEAN] mAP50    = {float(metrics.box.map50):.4f}")
        print(f"[CLEAN] mAP50-95 = {float(metrics.box.map):.4f}")
        print(f"[CLEAN] Precision= {float(metrics.box.mp):.4f}")
        print(f"[CLEAN] Recall   = {float(metrics.box.mr):.4f}")
        print("=====================================\n")

    evaluate_asr_gt(
        model=model,
        generator_path=args.generator,
        data_yaml=args.data,
        nc=args.nc,
        source_class=args.source,
        target_class=args.target,
        patch_size=args.patch_size,
        epsilon=args.epsilon,
        conf=args.conf,
        iou_thr=args.iou,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        save_dir=args.save_dir,
        save_debug=args.save_debug,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
