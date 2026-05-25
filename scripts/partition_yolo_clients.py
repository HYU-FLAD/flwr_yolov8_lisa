from pathlib import Path
import argparse
import random
import shutil
import yaml
from collections import Counter, defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def read_classes(label_file: Path):
    if not label_file.exists():
        return []

    classes = []
    for line in label_file.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 5:
            classes.append(int(float(parts[0])))

    return classes


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    if mode == "hardlink":
        try:
            import os
            os.link(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def copy_label_or_create_empty(src_label: Path, dst_label: Path, mode: str):
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if dst_label.exists():
        dst_label.unlink()

    if src_label.exists():
        copy_or_link(src_label, dst_label, mode)
    else:
        # 진짜 background image인 경우만 빈 label 생성
        dst_label.write_text("")


def resolve_path(root: Path, value: str):
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p


def collect_images(img_root: Path):
    images = []
    for ext in IMG_EXTS:
        images.extend(img_root.rglob(f"*{ext}"))
    return sorted(images)


def label_for_image(img: Path, img_root: Path, label_root: Path) -> Path:
    """
    핵심:
    images/train/dayTrain/.../xxx.jpg
    -> labels/train/dayTrain/.../xxx.txt
    """
    rel = img.relative_to(img_root)
    return label_root / rel.with_suffix(".txt")


def bucket_key(img: Path, img_root: Path, label_root: Path):
    lab = label_for_image(img, img_root, label_root)
    cs = read_classes(lab)

    if not cs:
        return "background"

    uniq = set(cs)
    if len(uniq) == 1:
        return f"class_{next(iter(uniq))}"

    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num-clients", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="hardlink")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text())

    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root

    train_img_root = resolve_path(root, cfg["train"])
    val_img_root = resolve_path(root, cfg["val"])

    test_value = cfg.get("test", cfg.get("eval", cfg["val"]))
    eval_img_root = resolve_path(root, test_value)

    train_label_root = root / "labels" / "train"
    val_label_root = root / "labels" / "val"
    eval_label_root = root / "labels" / "eval"

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_images = collect_images(train_img_root)

    if not train_images:
        raise RuntimeError(f"No train images found: {train_img_root}")

    print(f"[SOURCE] train_img_root={train_img_root}")
    print(f"[SOURCE] train_label_root={train_label_root}")
    print(f"[SOURCE] train_images={len(train_images)}")

    buckets = defaultdict(list)
    missing_labels = 0
    nonempty_labels = 0
    class_counter = Counter()

    for img in train_images:
        lab = label_for_image(img, train_img_root, train_label_root)
        cs = read_classes(lab)

        if not lab.exists():
            missing_labels += 1
        if cs:
            nonempty_labels += 1
            class_counter.update(cs)

        buckets[bucket_key(img, train_img_root, train_label_root)].append(img)

    print(f"[SOURCE] missing_label_files={missing_labels}")
    print(f"[SOURCE] nonempty_label_files={nonempty_labels}")
    print(f"[SOURCE] class_counts={dict(class_counter)}")

    rng = random.Random(args.seed)
    client_imgs = [[] for _ in range(args.num_clients)]

    # bucket별 round-robin 분배
    for key, imgs in sorted(buckets.items()):
        rng.shuffle(imgs)
        for i, img in enumerate(imgs):
            client_imgs[i % args.num_clients].append(img)
        print(f"[BUCKET] {key}: {len(imgs)}")

    for cid, imgs in enumerate(client_imgs):
        cdir = out_root / f"client_{cid}"

        # train split
        for img in imgs:
            rel = img.relative_to(train_img_root)
            lab = label_for_image(img, train_img_root, train_label_root)

            dst_img = cdir / "images" / "train" / rel
            dst_lab = cdir / "labels" / "train" / rel.with_suffix(".txt")

            copy_or_link(img, dst_img, args.mode)
            copy_label_or_create_empty(lab, dst_lab, args.mode)

        # shared val/eval
        for split_name, img_root, label_root in [
            ("val", val_img_root, val_label_root),
            ("eval", eval_img_root, eval_label_root),
        ]:
            split_images = collect_images(img_root)

            for img in split_images:
                rel = img.relative_to(img_root)
                lab = label_for_image(img, img_root, label_root)

                dst_img = cdir / "images" / split_name / rel
                dst_lab = cdir / "labels" / split_name / rel.with_suffix(".txt")

                copy_or_link(img, dst_img, args.mode)
                copy_label_or_create_empty(lab, dst_lab, args.mode)

        client_yaml = {
            "path": str(cdir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/eval",
            "nc": int(cfg["nc"]),
            "names": cfg["names"],
        }

        (cdir / "data.yaml").write_text(
            yaml.safe_dump(client_yaml, sort_keys=False, allow_unicode=True)
        )

        cnt = Counter()
        total_label_files = 0
        nonempty = 0
        empty = 0

        for lab in (cdir / "labels" / "train").rglob("*.txt"):
            total_label_files += 1
            cs = read_classes(lab)
            if cs:
                nonempty += 1
                cnt.update(cs)
            else:
                empty += 1

        print(
            f"[CLIENT {cid}] "
            f"train_images={len(imgs)}, "
            f"label_files={total_label_files}, "
            f"nonempty={nonempty}, "
            f"empty={empty}, "
            f"class_counts={dict(cnt)}"
        )

    print(f"[DONE] partition root: {out_root}")


if __name__ == "__main__":
    main()
