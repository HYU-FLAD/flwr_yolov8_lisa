from pathlib import Path
from collections import Counter
import argparse
import os
import random
import shutil
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default="datas/lisa_yolo_full/data.yaml")
    ap.add_argument("--out-root", default="datas/lisa_yolo_full/client_isolated_100")
    ap.add_argument("--num-clients", type=int, default=100)
    ap.add_argument("--source-name", default="stop")
    ap.add_argument("--target-name", default="go")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    data = yaml.safe_load(data_yaml.read_text())

    root = Path(data["path"])
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()

    names = data["names"]
    nc = int(data["nc"])

    if args.source_name not in names:
        raise ValueError(f"source-name {args.source_name} not in names: {names}")
    if args.target_name not in names:
        raise ValueError(f"target-name {args.target_name} not in names: {names}")

    source_id = names.index(args.source_name)
    target_id = names.index(args.target_name)

    train_img_dir = root / data["train"]
    train_label_dir = Path(str(train_img_dir).replace("/images/", "/labels/"))

    val_path = root / data["val"]
    test_path = root / data.get("test", "images/eval")

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing train image dir: {train_img_dir}")
    if not train_label_dir.exists():
        raise FileNotFoundError(f"Missing train label dir: {train_label_dir}")

    records = []
    skipped_missing = 0
    skipped_empty = 0

    for img in sorted(train_img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue

        rel = img.relative_to(train_img_dir)
        label = train_label_dir / rel.with_suffix(".txt")

        if not label.exists():
            skipped_missing += 1
            continue

        classes, lines = parse_label(label)
        if not lines or not classes:
            skipped_empty += 1
            continue

        records.append({
            "img": img,
            "label": label,
            "rel": rel,
            "classes": classes,
            "counter": Counter(classes),
        })

    if not records:
        raise RuntimeError("No valid train image-label pairs found. Do not partition broken dataset.")

    rng = random.Random(args.seed)

    # source class 포함 record를 먼저 분배해서 attacker가 stop을 받을 확률을 높임
    source_records = [r for r in records if r["counter"][source_id] > 0]
    other_records = [r for r in records if r["counter"][source_id] == 0]

    rng.shuffle(source_records)
    rng.shuffle(other_records)

    buckets = [[] for _ in range(args.num_clients)]
    bucket_cls = [Counter() for _ in range(args.num_clients)]

    for i, rec in enumerate(source_records):
        cid = i % args.num_clients
        buckets[cid].append(rec)
        bucket_cls[cid].update(rec["classes"])

    # 나머지는 이미지 수와 instance 수가 적은 client에 배분
    for rec in other_records:
        cid = min(
            range(args.num_clients),
            key=lambda i: (len(buckets[i]), sum(bucket_cls[i].values()))
        )
        buckets[cid].append(rec)
        bucket_cls[cid].update(rec["classes"])

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[PARTITION]")
    print(f"records: {len(records)}")
    print(f"source_id: {source_id} ({args.source_name})")
    print(f"target_id: {target_id} ({args.target_name})")
    print(f"skipped_missing: {skipped_missing}")
    print(f"skipped_empty: {skipped_empty}")

    bad = []
    no_source = []

    for cid, bucket in enumerate(buckets):
        cdir = out_root / f"client_{cid}"
        img_root = cdir / "images" / "train"
        lab_root = cdir / "labels" / "train"

        c_counter = Counter()

        for rec in bucket:
            dst_img = img_root / rec["rel"]
            dst_lab = lab_root / rec["rel"].with_suffix(".txt")

            link_or_copy(rec["img"], dst_img, args.mode)
            link_or_copy(rec["label"], dst_lab, args.mode)

            c_counter.update(rec["classes"])

        client_yaml = {
            "path": str(cdir.resolve()),
            "train": "images/train",
            "val": str(val_path.resolve()),
            "test": str(test_path.resolve()),
            "nc": nc,
            "names": names,
        }

        (cdir / "data.yaml").write_text(
            yaml.safe_dump(client_yaml, sort_keys=False, allow_unicode=True)
        )

        total_inst = sum(c_counter.values())
        print(
            f"client_{cid}: images={len(bucket)}, instances={total_inst}, "
            f"source={c_counter[source_id]}, target={c_counter[target_id]}"
        )

        if len(bucket) == 0 or total_inst == 0:
            bad.append(cid)
        if c_counter[source_id] == 0:
            no_source.append(cid)

    print("\n[VERIFY]")
    print("bad_clients:", bad, "count=", len(bad))
    print("no_source_clients:", no_source[:30], "count=", len(no_source))

    if bad:
        raise RuntimeError("Some clients have no valid labeled images.")

    print("[DONE] partition written:", out_root.resolve())

if __name__ == "__main__":
    main()
