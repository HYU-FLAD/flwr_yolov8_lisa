from pathlib import Path
import zipfile
import random
import shutil
import yaml
from collections import Counter, defaultdict
import argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def norm_stem(p: Path) -> str:
    return p.stem.strip()

def read_label_classes(label_path: Path):
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            classes.append(int(float(parts[0])))
    return classes

def safe_link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="datas/handmade_yolo")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--eval-ratio", type=float, default=0.15)
    ap.add_argument("--copy-mode", choices=["copy"], default="copy")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    out = Path(args.out)
    extract_root = out / "_raw_extract"

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_root)

    # 실제 root 탐색
    roots = [p for p in extract_root.iterdir() if p.is_dir()]
    raw = roots[0] if len(roots) == 1 else extract_root

    classes_file = raw / "classes.txt"
    if classes_file.exists():
        names = [x.strip() for x in classes_file.read_text(errors="ignore").splitlines() if x.strip()]
    else:
        names = ["red", "green"]

    if names != ["red", "green"]:
        print(f"[WARN] classes.txt names={names}. Expected ['red','green'].")

    # 모든 label은 raw 바로 아래에 있고, 이미지는 하위 폴더에 있음
    label_by_stem = {p.stem: p for p in raw.glob("*.txt") if p.name != "classes.txt"}

    image_paths = []
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            image_paths.append(p)

    pairs = []
    missing_labels = []
    for img in sorted(image_paths):
        stem = img.stem
        lab = label_by_stem.get(stem)
        if lab is None:
            missing_labels.append(str(img))
            continue
        pairs.append((img, lab))

    if missing_labels:
        print(f"[WARN] images without labels: {len(missing_labels)}")

    # stratification key: background/red/green/mixed
    buckets = defaultdict(list)
    cls_counter = Counter()
    for img, lab in pairs:
        cs = read_label_classes(lab)
        for c in cs:
            cls_counter[c] += 1
        if not cs:
            key = "background"
        elif set(cs) == {0}:
            key = "red"
        elif set(cs) == {1}:
            key = "green"
        else:
            key = "mixed"
        buckets[key].append((img, lab))

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "eval": []}

    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        train = items[:n_train]
        val = items[n_train:n_train+n_val]
        eval_ = items[n_train+n_val:]
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["eval"].extend(eval_)
        print(f"[SPLIT][{key}] total={n}, train={len(train)}, val={len(val)}, eval={len(eval_)}")

    for split, items in splits.items():
        rng.shuffle(items)
        for img, lab in items:
            img_dst = out / "images" / split / img.name
            lab_dst = out / "labels" / split / f"{img.stem}.txt"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lab_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_dst)
            # 빈 background label도 그대로 복사
            shutil.copy2(lab, lab_dst)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/eval",
        "nc": 2,
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    # summary
    for split in ["train", "val", "eval"]:
        imgs = list((out / "images" / split).glob("*"))
        labels = list((out / "labels" / split).glob("*.txt"))
        cnt = Counter()
        nonempty = 0
        for lab in labels:
            cs = read_label_classes(lab)
            if cs:
                nonempty += 1
            cnt.update(cs)
        print(f"[SUMMARY][{split}] images={len(imgs)}, labels={len(labels)}, nonempty={nonempty}, class_counts={dict(cnt)}")

    print(f"[DONE] data.yaml: {out / 'data.yaml'}")

if __name__ == "__main__":
    import os
    main()
