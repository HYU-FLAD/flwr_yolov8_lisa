from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import os
import random
import shutil
import yaml
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PREFERRED_CLASS_ORDER = [
    "go",
    "goForward",
    "goLeft",
    "stop",
    "stopLeft",
    "warning",
    "warningLeft",
]

def sniff_csv(path: Path):
    raw = path.read_text(errors="ignore")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"
    rows = []
    with path.open("r", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    return rows

def norm_col(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

def find_col(columns, candidates):
    normalized = {norm_col(c): c for c in columns}
    for cand in candidates:
        cand_n = norm_col(cand)
        if cand_n in normalized:
            return normalized[cand_n]
    for c in columns:
        n = norm_col(c)
        for cand in candidates:
            if norm_col(cand) in n:
                return c
    return None

def detect_columns(rows, csv_path: Path):
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    cols = list(rows[0].keys())

    filename_col = find_col(cols, [
        "Filename",
        "file",
        "image",
        "image file",
        "frame",
    ])

    class_col = find_col(cols, [
        "Annotation tag",
        "annotation",
        "class",
        "label",
        "tag",
    ])

    x1_col = find_col(cols, [
        "Upper left corner X",
        "upper left x",
        "x1",
        "xmin",
        "left",
    ])
    y1_col = find_col(cols, [
        "Upper left corner Y",
        "upper left y",
        "y1",
        "ymin",
        "top",
    ])
    x2_col = find_col(cols, [
        "Lower right corner X",
        "lower right x",
        "x2",
        "xmax",
        "right",
    ])
    y2_col = find_col(cols, [
        "Lower right corner Y",
        "lower right y",
        "y2",
        "ymax",
        "bottom",
    ])

    required = {
        "filename": filename_col,
        "class": class_col,
        "x1": x1_col,
        "y1": y1_col,
        "x2": x2_col,
        "y2": y2_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not detect columns {missing} in {csv_path}\n"
            f"Available columns: {cols}"
        )

    return required

def build_image_index(base: Path):
    images = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p.resolve())

    by_basename = defaultdict(list)
    by_rel = {}

    for img in images:
        rel = img.relative_to(base.resolve()).as_posix()
        by_rel[rel.lower()] = img
        by_basename[img.name.lower()].append(img)

    return images, by_rel, by_basename

def resolve_image(filename: str, base: Path, by_rel, by_basename):
    if filename is None:
        return None

    f = filename.strip().replace("\\", "/")
    f = f.lstrip("./")
    f_lower = f.lower()

    # 1) CSV 경로가 base 기준 상대경로와 정확히 맞는 경우
    if f_lower in by_rel:
        return by_rel[f_lower]

    # 2) base 하위 경로가 CSV filename으로 끝나는 경우
    suffix_matches = [p for rel, p in by_rel.items() if rel.endswith(f_lower)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 3) basename이 유일한 경우
    base_matches = by_basename.get(Path(f).name.lower(), [])
    if len(base_matches) == 1:
        return base_matches[0]

    # 4) basename이 여러 개면, CSV 경로 일부와 가장 잘 맞는 후보 선택
    if len(base_matches) > 1:
        scored = []
        f_parts = set(Path(f_lower).parts)
        for p in base_matches:
            rel = p.relative_to(base.resolve()).as_posix().lower()
            rel_parts = set(Path(rel).parts)
            score = len(f_parts & rel_parts)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]

    return None

def read_image_shape(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h

def valid_box(x1, y1, x2, y2, w, h, min_box_size):
    x1 = max(0.0, min(float(x1), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = x2 - x1
    bh = y2 - y1

    if bw < min_box_size or bh < min_box_size:
        return None

    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    nw = bw / w
    nh = bh / h

    vals = [xc, yc, nw, nh]
    if not all(0.0 <= v <= 1.0 for v in vals):
        return None

    return vals

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    # hardlink default
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def make_class_names(discovered):
    ordered = [c for c in PREFERRED_CLASS_ORDER if c in discovered]
    extras = sorted([c for c in discovered if c not in ordered])
    return ordered + extras

def split_records(records, names, train_ratio, val_ratio, eval_ratio, seed):
    if abs((train_ratio + val_ratio + eval_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/eval ratios must sum to 1.0")

    rng = random.Random(seed)

    n = len(records)
    targets = {
        "train": int(round(n * train_ratio)),
        "val": int(round(n * val_ratio)),
    }
    targets["eval"] = n - targets["train"] - targets["val"]

    cls_total = Counter()
    for rec in records:
        cls_total.update(set(rec["classes"]))

    def rarity_score(rec):
        score = 0.0
        for c in set(rec["classes"]):
            score += 1.0 / max(cls_total[c], 1)
        return score

    records = list(records)
    rng.shuffle(records)
    records.sort(key=rarity_score, reverse=True)

    buckets = {
        "train": [],
        "val": [],
        "eval": [],
    }
    bucket_cls = {
        "train": Counter(),
        "val": Counter(),
        "eval": Counter(),
    }

    ratio = {
        "train": train_ratio,
        "val": val_ratio,
        "eval": eval_ratio,
    }

    for rec in records:
        best_split = None
        best_score = None

        rec_classes = set(rec["classes"])

        for split in ["train", "val", "eval"]:
            if len(buckets[split]) >= targets[split]:
                continue

            # image count deficit
            img_deficit = targets[split] - len(buckets[split])

            # class distribution deficit
            cls_deficit = 0.0
            for c in rec_classes:
                desired = cls_total[c] * ratio[split]
                current = bucket_cls[split][c]
                cls_deficit += max(desired - current, 0.0)

            score = cls_deficit * 10.0 + img_deficit * 0.001

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        if best_split is None:
            best_split = min(buckets, key=lambda s: len(buckets[s]))

        buckets[best_split].append(rec)
        bucket_cls[best_split].update(rec["classes"])

    return buckets

def write_dataset(out: Path, base: Path, buckets, names, mode):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: i for i, name in enumerate(names)}

    stats = {}

    for split, recs in buckets.items():
        split_cls = Counter()
        split_instances = 0

        for rec in recs:
            img_path = rec["img"]
            rel = rec["rel"]

            dst_img = out / "images" / split / rel
            dst_label = out / "labels" / split / rel.with_suffix(".txt")

            link_or_copy(img_path, dst_img, mode)

            label_lines = []
            for obj in rec["objects"]:
                cid = class_to_id[obj["class"]]
                xc, yc, bw, bh = obj["xywh"]
                label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                split_cls[cid] += 1
                split_instances += 1

            if not label_lines:
                raise RuntimeError(f"Empty label would be written for {img_path}")

            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(label_lines) + "\n")

        stats[split] = {
            "images": len(recs),
            "instances": split_instances,
            "class_counts": {names[cid]: split_cls[cid] for cid in range(len(names))},
        }

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/eval",
        "nc": len(names),
        "names": names,
    }

    (out / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True)
    )
    (out / "stats.yaml").write_text(
        yaml.safe_dump(stats, sort_keys=False, allow_unicode=True)
    )

    return stats

def verify_output(out: Path, names):
    print("\n[VERIFY]")
    for split in ["train", "val", "eval"]:
        img_dir = out / "images" / split
        lab_dir = out / "labels" / split

        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        txts = list(lab_dir.rglob("*.txt"))

        nonempty = 0
        inst = 0
        cls_counter = Counter()

        for txt in txts:
            lines = [x.strip() for x in txt.read_text(errors="ignore").splitlines() if x.strip()]
            if lines:
                nonempty += 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    cls_counter[cid] += 1
                    inst += 1

        print(f"{split}: images={len(imgs)}, label_txt={len(txts)}, nonempty={nonempty}, instances={inst}")

        if len(imgs) == 0 or len(txts) == 0 or nonempty == 0 or inst == 0:
            raise RuntimeError(f"Broken YOLO split detected: {split}")

        for cid, name in enumerate(names):
            print(f"  {cid}: {name}: {cls_counter[cid]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="datas/lisa_base")
    ap.add_argument("--out", default="datas/lisa_yolo_full")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--eval-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-box-size", type=float, default=2.0)
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out = Path(args.out)

    if not base.exists():
        raise FileNotFoundError(f"Missing base dir: {base}")

    print("[SCAN] images...")
    images, by_rel, by_basename = build_image_index(base)
    print(f"[INFO] found images: {len(images)}")

    csv_files = sorted(base.rglob("frameAnnotationsBOX.csv"))
    if not csv_files:
        csv_files = sorted(base.rglob("frameAnnotations*.csv"))

    print(f"[INFO] found annotation CSVs: {len(csv_files)}")
    for p in csv_files[:10]:
        print(f"  - {p}")

    if not csv_files:
        raise FileNotFoundError("No frameAnnotations*.csv found under lisa_base")

    image_objects = defaultdict(list)
    image_shapes = {}
    unresolved = 0
    invalid_boxes = 0
    total_rows = 0

    for csv_path in csv_files:
        rows = sniff_csv(csv_path)
        if not rows:
            continue

        cols = detect_columns(rows, csv_path)

        for row in rows:
            total_rows += 1

            img_path = resolve_image(row[cols["filename"]], base, by_rel, by_basename)
            if img_path is None:
                unresolved += 1
                continue

            cls_name = str(row[cols["class"]]).strip()
            if not cls_name:
                continue

            if img_path not in image_shapes:
                shape = read_image_shape(img_path)
                if shape is None:
                    unresolved += 1
                    continue
                image_shapes[img_path] = shape

            w, h = image_shapes[img_path]

            try:
                box = valid_box(
                    row[cols["x1"]],
                    row[cols["y1"]],
                    row[cols["x2"]],
                    row[cols["y2"]],
                    w,
                    h,
                    args.min_box_size,
                )
            except Exception:
                box = None

            if box is None:
                invalid_boxes += 1
                continue

            image_objects[img_path].append({
                "class": cls_name,
                "xywh": box,
            })

    records = []
    discovered_classes = set()

    for img_path, objects in image_objects.items():
        if not objects:
            continue

        rel = img_path.relative_to(base)
        classes = [obj["class"] for obj in objects]
        discovered_classes.update(classes)

        records.append({
            "img": img_path,
            "rel": rel,
            "objects": objects,
            "classes": classes,
        })

    if not records:
        raise RuntimeError(
            "No valid labeled images were produced. "
            "Check LISA CSV path columns and image path mapping."
        )

    names = make_class_names(discovered_classes)

    print("\n[SUMMARY]")
    print(f"csv_rows: {total_rows}")
    print(f"valid_labeled_images: {len(records)}")
    print(f"unresolved_rows: {unresolved}")
    print(f"invalid_boxes: {invalid_boxes}")
    print(f"classes ({len(names)}):")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")

    buckets = split_records(
        records,
        names,
        args.train_ratio,
        args.val_ratio,
        args.eval_ratio,
        args.seed,
    )

    stats = write_dataset(out, base, buckets, names, args.mode)

    print("\n[DONE] YOLO dataset written")
    print(f"out: {out.resolve()}")
    print(f"data_yaml: {(out / 'data.yaml').resolve()}")

    print("\n[STATS]")
    for split, s in stats.items():
        print(f"{split}: images={s['images']}, instances={s['instances']}")
        for k, v in s["class_counts"].items():
            print(f"  {k}: {v}")

    verify_output(out, names)

if __name__ == "__main__":
    main()
