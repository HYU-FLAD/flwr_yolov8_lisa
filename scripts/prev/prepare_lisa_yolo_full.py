import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import yaml


RAW_ROOT = Path("datas/lisa_base")
OUT_ROOT = Path("datas/lisa_yolo_full")

SEED = 2026
USE_SYMLINK = True

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "eval": 0.15,
}


def find_column(fieldnames, candidates):
    lower_map = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of columns: {candidates}. Available={fieldnames}")


def open_csv_reader(csv_path):
    f = csv_path.open("r", newline="", encoding="utf-8", errors="ignore")

    reader = csv.DictReader(f, delimiter=";")
    if reader.fieldnames is None or len(reader.fieldnames) <= 1:
        f.seek(0)
        reader = csv.DictReader(f, delimiter=",")

    return f, reader


def build_image_index(raw_root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    by_name = defaultdict(list)

    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            by_name[p.name].append(p.resolve())

    return by_name


def resolve_image_path(filename: str, raw_root: Path, image_index):
    filename = str(filename).replace("\\", "/").strip()
    p = Path(filename)

    direct = raw_root / p
    if direct.exists():
        return direct.resolve()

    suffix_matches = []
    for paths in image_index.values():
        for img_path in paths:
            if str(img_path).replace("\\", "/").endswith(filename):
                suffix_matches.append(img_path)

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    basename_matches = image_index.get(p.name, [])
    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        print(f"[WARN] Ambiguous basename match for {filename}: {len(basename_matches)} candidates")

    return None


def safe_name_from_path(img_path: Path, raw_root: Path):
    rel = img_path.resolve().relative_to(raw_root.resolve())
    safe = "__".join(rel.parts)
    return safe


def read_annotations(raw_root: Path):
    image_index = build_image_index(raw_root)
    ann_files = sorted(raw_root.rglob("frameAnnotationsBOX.csv"))

    if not ann_files:
        raise FileNotFoundError(f"No frameAnnotationsBOX.csv found under {raw_root.resolve()}")

    records_by_image = defaultdict(list)
    class_counter = Counter()
    missing_images = 0
    bad_boxes = 0

    missing_log = OUT_ROOT / "missing_images.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with missing_log.open("w") as missing_f:
        for csv_path in ann_files:
            print(f"[INFO] Reading {csv_path}")

            f, reader = open_csv_reader(csv_path)
            with f:
                fieldnames = reader.fieldnames

                filename_col = find_column(fieldnames, ["Filename", "filename"])
                tag_col = find_column(fieldnames, ["Annotation tag", "Annotation Tag", "annotation tag"])

                x1_col = find_column(fieldnames, ["Upper left corner X", "upper left corner x"])
                y1_col = find_column(fieldnames, ["Upper left corner Y", "upper left corner y"])
                x2_col = find_column(fieldnames, ["Lower right corner X", "lower right corner x"])
                y2_col = find_column(fieldnames, ["Lower right corner Y", "lower right corner y"])

                for row in reader:
                    tag = str(row.get(tag_col, "")).strip()
                    if not tag:
                        continue

                    img_path = resolve_image_path(row[filename_col], raw_root, image_index)
                    if img_path is None:
                        missing_images += 1
                        missing_f.write(f"{csv_path}\t{row.get(filename_col, '')}\n")
                        continue

                    try:
                        x1 = float(row[x1_col])
                        y1 = float(row[y1_col])
                        x2 = float(row[x2_col])
                        y2 = float(row[y2_col])
                    except Exception:
                        bad_boxes += 1
                        continue

                    if x2 <= x1 or y2 <= y1:
                        bad_boxes += 1
                        continue

                    records_by_image[str(img_path)].append((tag, x1, y1, x2, y2))
                    class_counter[tag] += 1

    class_names = sorted(class_counter.keys())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print("\n[INFO] Original LISA classes preserved:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name} ({class_counter[name]})")

    print(f"\n[INFO] Images with labels: {len(records_by_image)}")
    print(f"[INFO] Missing images: {missing_images}")
    print(f"[INFO] Bad boxes: {bad_boxes}")
    print(f"[INFO] Missing image log: {missing_log}")

    return records_by_image, class_names, class_to_id, class_counter


def split_images(image_paths):
    rng = random.Random(SEED)
    image_paths = sorted(image_paths)
    rng.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train + n_val],
        "eval": image_paths[n_train + n_val:],
    }


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if USE_SYMLINK:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w - 1))
    y2 = max(0.0, min(y2, img_h - 1))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def write_dataset(records_by_image, class_names, class_to_id, class_counter, raw_root: Path, out_root: Path):
    if out_root.exists():
        print(f"[WARN] Output exists: {out_root}")
        print("[WARN] Existing files may be overwritten.")

    splits = split_images(list(records_by_image.keys()))
    split_counts = {}
    split_class_counts = {}

    for split, img_paths in splits.items():
        split_counts[split] = len(img_paths)
        split_class_counts[split] = Counter()

        img_out_dir = out_root / "images" / split
        lbl_out_dir = out_root / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path_str in img_paths:
            img_path = Path(img_path_str)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            safe_name = safe_name_from_path(img_path, raw_root)
            out_img = img_out_dir / safe_name
            out_lbl = lbl_out_dir / f"{Path(safe_name).stem}.txt"

            link_or_copy(img_path, out_img)

            lines = []
            for tag, x1, y1, x2, y2 in records_by_image[img_path_str]:
                yolo_box = convert_bbox_to_yolo(x1, y1, x2, y2, w, h)
                if yolo_box is None:
                    continue

                cls_id = class_to_id[tag]
                xc, yc, bw, bh = yolo_box
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                split_class_counts[split][tag] += 1

            with out_lbl.open("w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/eval",
        "nc": len(class_names),
        "names": class_names,
    }

    with (out_root / "data.yaml").open("w") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    class_map = {
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
        "counts_total": {name: int(class_counter[name]) for name in class_names},
        "counts_by_split": {
            split: {name: int(split_class_counts[split][name]) for name in class_names}
            for split in ["train", "val", "eval"]
        },
    }

    with (out_root / "class_map.yaml").open("w") as f:
        yaml.dump(class_map, f, sort_keys=False, allow_unicode=True)

    print("\n[DONE] Full-class LISA YOLO dataset created.")
    print("[INFO] Split counts:", split_counts)
    print(f"[INFO] data.yaml: {out_root / 'data.yaml'}")
    print(f"[INFO] class_map.yaml: {out_root / 'class_map.yaml'}")


def main():
    records_by_image, class_names, class_to_id, class_counter = read_annotations(RAW_ROOT)
    write_dataset(
        records_by_image=records_by_image,
        class_names=class_names,
        class_to_id=class_to_id,
        class_counter=class_counter,
        raw_root=RAW_ROOT,
        out_root=OUT_ROOT,
    )


if __name__ == "__main__":
    main()
