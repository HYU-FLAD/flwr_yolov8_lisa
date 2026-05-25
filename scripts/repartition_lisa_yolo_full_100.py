from pathlib import Path
from collections import Counter
import argparse
import os
import random
import shutil
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def resolve_path(base_yaml: Path, root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p

def parse_label(txt: Path):
    classes = []
    lines = [x.strip() for x in txt.read_text(errors="ignore").splitlines() if x.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(x) for x in parts[1:5]]
            if all(0.0 <= v <= 1.0 for v in vals):
                classes.append(cid)
        except Exception:
            continue
    return classes, lines

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return

    # hardlink default
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default="datas/lisa_yolo_full/data.yaml")
    ap.add_argument("--out-root", default="datas/lisa_yolo_full/client_isolated_100")
    ap.add_argument("--num-clients", type=int, default=100)
    ap.add_argument("--source-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    data = yaml.safe_load(data_yaml.read_text())

    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()

    train_img_dir = resolve_path(data_yaml, root, data["train"]).resolve()
    val_path = resolve_path(data_yaml, root, data["val"]).resolve()

    train_label_dir = Path(str(train_img_dir).replace("/images/", "/labels/")).resolve()

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing train image dir: {train_img_dir}")
    if not train_label_dir.exists():
        raise FileNotFoundError(f"Missing train label dir: {train_label_dir}")

    records = []
    missing_label = 0
    empty_label = 0
    invalid_label = 0

    for img in sorted(train_img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue

        rel = img.relative_to(train_img_dir)
        label = train_label_dir / rel.with_suffix(".txt")

        if not label.exists():
            missing_label += 1
            continue

        classes, lines = parse_label(label)
        if label.stat().st_size == 0 or not lines:
            empty_label += 1
            continue
        if not classes:
            invalid_label += 1
            continue

        records.append((img, label, rel, Counter(classes)))

    if not records:
        raise RuntimeError(
            "No valid image-label pairs found. "
            "datas/lisa_yolo_full itself is probably broken."
        )

    rng = random.Random(args.seed)

    source_records = [r for r in records if r[3][args.source_class] > 0]
    other_records = [r for r in records if r[3][args.source_class] == 0]

    rng.shuffle(source_records)
    rng.shuffle(other_records)

    buckets = [[] for _ in range(args.num_clients)]

    # 1) source class 포함 이미지 먼저 round-robin 배분
    for i, rec in enumerate(source_records):
        buckets[i % args.num_clients].append(rec)

    # 2) 나머지는 현재 이미지 수가 가장 적은 client에 배분
    for rec in other_records:
        idx = min(range(args.num_clients), key=lambda i: len(buckets[i]))
        buckets[idx].append(rec)

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names = data["names"]
    nc = int(data["nc"])

    total_written = 0
    global_cls = Counter()

    for cid, bucket in enumerate(buckets):
        cdir = out_root / f"client_{cid}"
        img_dst_root = cdir / "images" / "train"
        lab_dst_root = cdir / "labels" / "train"

        c_counter = Counter()

        for img, label, rel, counter in bucket:
            dst_img = img_dst_root / rel
            dst_lab = lab_dst_root / rel.with_suffix(".txt")

            link_or_copy(img, dst_img, args.mode)
            link_or_copy(label, dst_lab, args.mode)

            c_counter.update(counter)
            global_cls.update(counter)
            total_written += 1

        client_yaml = {
            "path": str(cdir.resolve()),
            "train": "images/train",
            "val": str(val_path),
            "nc": nc,
            "names": names,
        }

        if "test" in data:
            test_path = resolve_path(data_yaml, root, data["test"]).resolve()
            client_yaml["test"] = str(test_path)

        (cdir / "data.yaml").write_text(
            yaml.safe_dump(client_yaml, sort_keys=False, allow_unicode=True)
        )

        print(
            f"client_{cid}: images={len(bucket)}, "
            f"instances={sum(c_counter.values())}, "
            f"source_{args.source_class}={c_counter[args.source_class]}"
        )

    print("\n[DONE]")
    print("valid_pairs:", len(records))
    print("written_images:", total_written)
    print("missing_label:", missing_label)
    print("empty_label:", empty_label)
    print("invalid_label:", invalid_label)
    print("global_class_distribution:")
    for cid, name in enumerate(names):
        print(f"  {cid}: {name}: {global_cls[cid]}")

if __name__ == "__main__":
    main()
