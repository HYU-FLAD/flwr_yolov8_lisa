#!/usr/bin/env python3
"""
Preprocess a small YOLO-format red/green zip dataset for the nc=2 branch.

The output is compatible with:
  - pyproject.toml num-classes = 2
  - yolov8n_custom.yaml nc: 2
  - fl_yolo_backdoor client/server nc=2 branch

Modes:
  round_robin: split train images once across clients.
  resample / resample_augmented: each client samples a different subset from the global
    train pool. Default client-local augmentation is 0.0, so generated labels are not
    distorted unless explicitly requested.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    stem: str
    class_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path
    class_ids: Tuple[int, ...]


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def sanitize_stem(text: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text]
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "sample"


def read_runtime_nc_from_pyproject(pyproject_path: Path) -> int | None:
    if not pyproject_path.exists() or tomllib is None:
        return None
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return int(data["tool"]["flwr"]["app"]["config"].get("num-classes"))
    except Exception:
        return None


def read_classes(classes_file: Path, yaml_nc: int) -> List[str]:
    names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"No class names found in {classes_file}")
    if len(names) != yaml_nc:
        raise ValueError(
            f"This nc=2 branch expects exactly {yaml_nc} class names, but classes.txt has {len(names)}: {names}. "
            "For red/green data this should be ['red', 'green']."
        )
    return names


def parse_yolo_label(label_path: Path) -> Tuple[int, ...]:
    class_ids: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return tuple(sorted(set(class_ids)))


def validate_label_file(label_path: Path, yaml_nc: int) -> None:
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line: {label_path}:{line_no}: {line}")
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric label: {label_path}:{line_no}: {line}") from exc
        if cls_id < 0 or cls_id >= yaml_nc:
            raise ValueError(f"Class id {cls_id} in {label_path}:{line_no} is outside nc={yaml_nc}")
        if bw <= 0 or bh <= 0:
            raise ValueError(f"Non-positive bbox size: {label_path}:{line_no}: {line}")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            raise ValueError(f"BBox is not normalized: {label_path}:{line_no}: {line}")


def find_samples(extract_root: Path, yaml_nc: int) -> List[Sample]:
    txt_files = [p for p in extract_root.rglob("*.txt") if p.name != "classes.txt"]

    img_index: Dict[str, Path] = {}
    for img in extract_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            img_index[img.stem] = img

    samples: List[Sample] = []
    missing_images = []
    for lbl in sorted(txt_files):
        img = img_index.get(lbl.stem)
        if img is None:
            missing_images.append(lbl.name)
            continue
        validate_label_file(lbl, yaml_nc)
        class_ids = parse_yolo_label(lbl)
        if not class_ids:
            continue
        samples.append(Sample(img, lbl, lbl.stem, class_ids))

    if missing_images:
        print(f"[WARN] labels without matching image skipped: {len(missing_images)}")
    if not samples:
        raise RuntimeError("No valid image/label pairs found. Check zip structure.")
    return samples


def stratified_split(samples: Sequence[Sample], train_ratio: float, val_ratio: float, eval_ratio: float, seed: int):
    if abs((train_ratio + val_ratio + eval_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/eval ratios must sum to 1.0")

    rng = random.Random(seed)
    buckets: Dict[Tuple[int, ...], List[Sample]] = {}
    for s in samples:
        buckets.setdefault(s.class_ids, []).append(s)

    splits = {"train": [], "val": [], "eval": []}
    for _, bucket in sorted(buckets.items(), key=lambda kv: kv[0]):
        bucket = list(bucket)
        rng.shuffle(bucket)
        n = len(bucket)
        if n >= 3:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
        else:
            n_train = max(1, n - 1)
            n_val = 0
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_eval = n - n_train - n_val
        if n >= 3 and n_eval == 0:
            if n_train > 1:
                n_train -= 1
                n_eval = 1
            elif n_val > 1:
                n_val -= 1
                n_eval = 1
        splits["train"].extend(bucket[:n_train])
        splits["val"].extend(bucket[n_train:n_train + n_val])
        splits["eval"].extend(bucket[n_train + n_val:])

    for v in splits.values():
        rng.shuffle(v)
    return splits


def write_pair(sample: Sample, out_root: Path, split: str, prefix: str = "") -> Pair:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_stem(f"{prefix}{sample.stem}")
    out_img = img_dir / f"{base}{sample.image_path.suffix.lower()}"
    out_lbl = lbl_dir / f"{base}.txt"
    shutil.copy2(sample.image_path, out_img)
    shutil.copy2(sample.label_path, out_lbl)
    return Pair(out_img, out_lbl, sample.class_ids)


def read_label_lines(label_path: Path) -> List[str]:
    return [line.strip() for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def clip_yolo_box(xc: float, yc: float, bw: float, bh: float):
    x1 = max(0.0, xc - bw / 2.0)
    y1 = max(0.0, yc - bh / 2.0)
    x2 = min(1.0, xc + bw / 2.0)
    y2 = min(1.0, yc + bh / 2.0)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 1e-6 or nh <= 1e-6:
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, nw, nh


def augment_image_and_labels(img: np.ndarray, label_lines: List[str], rng: random.Random):
    out = img.copy()
    parsed = []
    for line in label_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        parsed.append([cls_id, xc, yc, bw, bh])
    if not parsed:
        return out, []

    # Conservative geometric transform. No horizontal flip by default because traffic-light
    # labels are color-driven and we do not need strong spatial augmentation here.
    h, w = out.shape[:2]
    angle = rng.uniform(-2.0, 2.0)
    scale = rng.uniform(0.98, 1.02)
    tx = rng.uniform(-0.01, 0.01) * w
    ty = rng.uniform(-0.01, 0.01) * h
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    labels = []
    for cls_id, xc, yc, bw, bh in parsed:
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        warped = np.hstack([corners, np.ones((4, 1), dtype=np.float32)]) @ matrix.T
        nx1, ny1 = warped[:, 0].min() / w, warped[:, 1].min() / h
        nx2, ny2 = warped[:, 0].max() / w, warped[:, 1].max() / h
        clipped = clip_yolo_box((nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0, nx2 - nx1, ny2 - ny1)
        if clipped is None:
            continue
        nxc, nyc, nbw, nbh = clipped
        labels.append(f"{cls_id} {nxc:.6f} {nyc:.6f} {nbw:.6f} {nbh:.6f}")

    if not labels:
        return out, []

    alpha = rng.uniform(0.95, 1.05)
    beta = rng.uniform(-5, 5)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    return out, labels


def augment_train_split(out_root: Path, copies: int, seed: int) -> int:
    if copies <= 0:
        return 0
    rng = random.Random(seed + 999)
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    created = 0
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        labels = read_label_lines(lbl_path)
        for k in range(copies):
            aug_img, aug_lbl = augment_image_and_labels(img, labels, rng)
            if not aug_lbl:
                continue
            out_img = img_dir / f"{img_path.stem}_aug{k + 1}{img_path.suffix.lower()}"
            out_lbl = lbl_dir / f"{img_path.stem}_aug{k + 1}.txt"
            cv2.imwrite(str(out_img), aug_img)
            out_lbl.write_text("\n".join(aug_lbl) + "\n", encoding="utf-8")
            created += 1
    return created


def write_data_yaml(path: Path, dataset_root: Path, names: List[str], train: str, val: str, test: str | None = None):
    data = {
        "path": str(dataset_root.resolve()),
        "train": train,
        "val": val,
        "names": {i: name for i, name in enumerate(names)},
        "nc": len(names),
    }
    if test is not None:
        data["test"] = test
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def collect_train_pairs(out_root: Path) -> List[Pair]:
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    pairs = []
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            class_ids = parse_yolo_label(lbl_path)
            if class_ids:
                pairs.append(Pair(img_path, lbl_path, class_ids))
    if not pairs:
        raise RuntimeError("No train image/label pairs found.")
    return pairs


def write_client_yamls(client_root: Path, out_root: Path, names: List[str], total_clients: int):
    for cid in range(total_clients):
        cdir = client_root / f"client_{cid}"
        (cdir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (cdir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        write_data_yaml(
            cdir / "data.yaml",
            cdir,
            names,
            train="images/train",
            val=str((out_root / "images" / "val").resolve()),
            test=str((out_root / "images" / "eval").resolve()),
        )


def build_client_round_robin(out_root: Path, names: List[str], total_clients: int, seed: int, link_mode: str):
    client_root = out_root / f"client_isolated_{total_clients}"
    safe_rmtree(client_root)
    client_root.mkdir(parents=True, exist_ok=True)
    pairs = collect_train_pairs(out_root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    counts = {i: 0 for i in range(total_clients)}
    for idx, pair in enumerate(pairs):
        cid = idx % total_clients
        dst_img = client_root / f"client_{cid}" / "images" / "train" / pair.image_path.name
        dst_lbl = client_root / f"client_{cid}" / "labels" / "train" / pair.label_path.name
        link_or_copy(pair.image_path, dst_img, link_mode)
        link_or_copy(pair.label_path, dst_lbl, link_mode)
        counts[cid] += 1
    write_client_yamls(client_root, out_root, names, total_clients)
    return counts


def build_client_resample(
    out_root: Path,
    names: List[str],
    total_clients: int,
    seed: int,
    client_target_images: int,
    link_mode: str,
    client_local_aug_prob: float,
):
    if client_target_images <= 0:
        raise ValueError("--client-target-images must be > 0")
    if not 0.0 <= client_local_aug_prob <= 1.0:
        raise ValueError("--client-local-aug-prob must be in [0, 1]")

    client_root = out_root / f"client_isolated_{total_clients}"
    safe_rmtree(client_root)
    client_root.mkdir(parents=True, exist_ok=True)
    pairs = collect_train_pairs(out_root)
    counts = {i: 0 for i in range(total_clients)}

    for cid in range(total_clients):
        rng = random.Random(seed + cid * 1_000_003)
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        pool_idx = 0
        c_img_dir = client_root / f"client_{cid}" / "images" / "train"
        c_lbl_dir = client_root / f"client_{cid}" / "labels" / "train"
        c_img_dir.mkdir(parents=True, exist_ok=True)
        c_lbl_dir.mkdir(parents=True, exist_ok=True)

        while counts[cid] < client_target_images:
            if pool_idx >= len(shuffled):
                rng.shuffle(shuffled)
                pool_idx = 0
            pair = shuffled[pool_idx]
            pool_idx += 1

            base = sanitize_stem(f"c{cid:03d}_{counts[cid]:04d}_{pair.image_path.stem}")
            dst_img = c_img_dir / f"{base}{pair.image_path.suffix.lower()}"
            dst_lbl = c_lbl_dir / f"{base}.txt"

            if client_local_aug_prob > 0 and rng.random() < client_local_aug_prob:
                img = cv2.imread(str(pair.image_path))
                if img is None:
                    continue
                label_lines = read_label_lines(pair.label_path)
                aug_img, aug_lbl = augment_image_and_labels(img, label_lines, rng)
                if not aug_lbl:
                    continue
                cv2.imwrite(str(dst_img), aug_img)
                dst_lbl.write_text("\n".join(aug_lbl) + "\n", encoding="utf-8")
            else:
                link_or_copy(pair.image_path, dst_img, link_mode)
                link_or_copy(pair.label_path, dst_lbl, link_mode)
            counts[cid] += 1

    write_client_yamls(client_root, out_root, names, total_clients)
    return counts


def summarize_labels(labels_dir: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for lbl in labels_dir.rglob("*.txt"):
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                counts[cls_id] = counts.get(cls_id, 0) + 1
    return dict(sorted(counts.items()))


def remove_label_caches(root: Path):
    for cache in root.rglob("*.cache"):
        try:
            cache.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Input zip path, e.g. ./datas/image-labeled.zip")
    parser.add_argument("--out", default="datas/lisa_yolo", help="Output dataset root")
    parser.add_argument("--total-clients", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--aug-copies", type=int, default=1, help="Global train augmentation copies")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--yaml-nc", type=int, default=None, help="Defaults to pyproject num-classes; must be 2 here")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument(
        "--client-sampling-mode",
        choices=["round_robin", "resample", "resample_augmented"],
        default="resample",
    )
    parser.add_argument("--client-target-images", type=int, default=30)
    parser.add_argument("--client-local-aug-prob", type=float, default=0.0)
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()

    runtime_nc = read_runtime_nc_from_pyproject(Path(args.pyproject)) or 2
    yaml_nc = args.yaml_nc if args.yaml_nc is not None else runtime_nc
    if runtime_nc != 2 or yaml_nc != 2:
        raise SystemExit(
            f"[FATAL] This nc=2 branch requires runtime_nc=2 and yaml_nc=2. "
            f"Got runtime_nc={runtime_nc}, yaml_nc={yaml_nc}."
        )

    zip_path = Path(args.zip)
    out_root = Path(args.out)
    tmp_root = out_root.parent / f".{out_root.name}_extract_tmp"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    safe_rmtree(out_root)
    safe_rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_root)

    classes_candidates = list(tmp_root.rglob("classes.txt"))
    if not classes_candidates:
        raise FileNotFoundError("classes.txt not found in zip")
    names = read_classes(classes_candidates[0], yaml_nc=yaml_nc)
    samples = find_samples(tmp_root, yaml_nc=yaml_nc)
    splits = stratified_split(samples, args.train_ratio, args.val_ratio, args.eval_ratio, args.seed)

    for split, split_samples in splits.items():
        for sample in split_samples:
            write_pair(sample, out_root, split)

    aug_created = augment_train_split(out_root, args.aug_copies, args.seed)
    write_data_yaml(out_root / "data.yaml", out_root, names, train="images/train", val="images/val", test="images/eval")

    if args.client_sampling_mode == "round_robin":
        client_counts = build_client_round_robin(out_root, names, args.total_clients, args.seed, args.link_mode)
    else:
        client_counts = build_client_resample(
            out_root,
            names,
            args.total_clients,
            args.seed,
            args.client_target_images,
            args.link_mode,
            args.client_local_aug_prob,
        )

    remove_label_caches(out_root)

    print("\n[OK] nc=2 preprocessing completed")
    print(f"- output root      : {out_root.resolve()}")
    print(f"- runtime nc       : {runtime_nc}")
    print(f"- yaml nc          : {yaml_nc}")
    print(f"- classes          : {names}")
    print(f"- original pairs   : {len(samples)}")
    print(f"- split originals  : train={len(splits['train'])}, val={len(splits['val'])}, eval={len(splits['eval'])}")
    print(f"- augmented train  : +{aug_created}")
    print(f"- final images     : train={len(list((out_root / 'images/train').glob('*')))}, val={len(list((out_root / 'images/val').glob('*')))}, eval={len(list((out_root / 'images/eval').glob('*')))}")
    print(f"- train label count: {summarize_labels(out_root / 'labels/train')}")
    print(f"- val label count  : {summarize_labels(out_root / 'labels/val')}")
    print(f"- eval label count : {summarize_labels(out_root / 'labels/eval')}")
    nonempty = sum(1 for v in client_counts.values() if v > 0)
    print(f"- clients          : {nonempty}/{args.total_clients} non-empty; min={min(client_counts.values())}, max={max(client_counts.values())}")
    print(f"- client mode      : {args.client_sampling_mode}")
    print(f"- client target    : {args.client_target_images if args.client_sampling_mode != 'round_robin' else 'n/a'}")
    print(f"- client aug prob  : {args.client_local_aug_prob}")
    print(f"- global yaml      : {out_root / 'data.yaml'}")
    print(f"- client yaml ex   : {out_root / f'client_isolated_{args.total_clients}' / 'client_0' / 'data.yaml'}")

    if not args.keep_extracted:
        safe_rmtree(tmp_root)


if __name__ == "__main__":
    main()
