from pathlib import Path
import argparse
import random
import shutil
import yaml
from collections import Counter, defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


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


def label_for_image(img: Path, dataset_root: Path):
    rel = img.relative_to(dataset_root)
    parts = list(rel.parts)

    if "images" not in parts:
        raise ValueError(f"'images' not found in image path: {rel}")

    idx = parts.index("images")
    parts[idx] = "labels"

    return dataset_root / Path(*parts).with_suffix(".txt")


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
            try:
                classes.append(int(float(parts[0])))
            except Exception:
                pass

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


def copy_label_or_empty(src_label: Path, dst_label: Path, mode: str):
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if dst_label.exists():
        dst_label.unlink()

    if src_label.exists():
        copy_or_link(src_label, dst_label, mode)
    else:
        dst_label.write_text("")


def bucket_key(img: Path, dataset_root: Path):
    lab = label_for_image(img, dataset_root)
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
    ap.add_argument("--num-clients", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="hardlink")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text())

    dataset_root = Path(cfg.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()

    train_img_root = resolve_path(dataset_root, cfg["train"])
    val_img_root = resolve_path(dataset_root, cfg["val"])

    test_value = cfg.get("test", cfg.get("eval", cfg["val"]))
    test_img_root = resolve_path(dataset_root, test_value)

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_images = collect_images(train_img_root)
    if not train_images:
        raise RuntimeError(f"No train images found: {train_img_root}")

    print(f"[SOURCE] dataset_root={dataset_root}")
    print(f"[SOURCE] train_img_root={train_img_root}")
    print(f"[SOURCE] val_img_root={val_img_root}")
    print(f"[SOURCE] train_images={len(train_images)}")

    buckets = defaultdict(list)
    missing_labels = 0
    nonempty_labels = 0
    class_counter = Counter()

    for img in train_images:
        lab = label_for_image(img, dataset_root)
        cs = read_classes(lab)

        if not lab.exists():
            missing_labels += 1
        if cs:
            nonempty_labels += 1
            class_counter.update(cs)

        buckets[bucket_key(img, dataset_root)].append(img)

    print(f"[SOURCE] missing_label_files={missing_labels}")
    print(f"[SOURCE] nonempty_label_files={nonempty_labels}")
    print(f"[SOURCE] class_counts_top20={class_counter.most_common(20)}")

    rng = random.Random(args.seed)
    client_imgs = [[] for _ in range(args.num_clients)]

    for key, imgs in sorted(buckets.items()):
        rng.shuffle(imgs)
        for i, img in enumerate(imgs):
            client_imgs[i % args.num_clients].append(img)
        print(f"[BUCKET] {key}: {len(imgs)}")

    for cid, imgs in enumerate(client_imgs):
        cdir = out_root / f"client_{cid}"
        cdir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            rel = img.relative_to(dataset_root)
            lab = label_for_image(img, dataset_root)

            dst_img = cdir / rel

            dst_parts = list(rel.parts)
            idx = dst_parts.index("images")
            dst_parts[idx] = "labels"
            dst_lab = cdir / Path(*dst_parts).with_suffix(".txt")

            copy_or_link(img, dst_img, args.mode)
            copy_label_or_empty(lab, dst_lab, args.mode)

        for split_name, img_root in [("val", val_img_root), ("test", test_img_root)]:
            split_imgs = collect_images(img_root)

            for img in split_imgs:
                rel = img.relative_to(dataset_root)
                lab = label_for_image(img, dataset_root)

                dst_img = cdir / rel

                dst_parts = list(rel.parts)
                idx = dst_parts.index("images")
                dst_parts[idx] = "labels"
                dst_lab = cdir / Path(*dst_parts).with_suffix(".txt")

                copy_or_link(img, dst_img, args.mode)
                copy_label_or_empty(lab, dst_lab, args.mode)

        client_yaml = {
            "path": str(cdir.resolve()),
            "train": cfg["train"],
            "val": cfg["val"],
            "nc": int(cfg["nc"]),
            "names": cfg["names"],
        }

        if "test" in cfg:
            client_yaml["test"] = cfg["test"]

        (cdir / "data.yaml").write_text(
            yaml.safe_dump(client_yaml, sort_keys=False, allow_unicode=True)
        )

        train_labels = []
        train_rel = Path(cfg["train"])
        train_label_rel = Path(*["labels" if x == "images" else x for x in train_rel.parts])
        train_label_root = cdir / train_label_rel

        if train_label_root.exists():
            train_labels = list(train_label_root.rglob("*.txt"))

        c_cnt = Counter()
        nonempty = 0
        for lab in train_labels:
            cs = read_classes(lab)
            if cs:
                nonempty += 1
                c_cnt.update(cs)

        print(
            f"[CLIENT {cid}] "
            f"train_images={len(imgs)}, "
            f"train_labels={len(train_labels)}, "
            f"nonempty={nonempty}, "
            f"class_counts_top10={c_cnt.most_common(10)}"
        )

    missing = []
    for cid in range(args.num_clients):
        if not (out_root / f"client_{cid}" / "data.yaml").exists():
            missing.append(cid)

    if missing:
        raise RuntimeError(f"[FATAL] missing client data.yaml: {missing}")

    print(f"[CHECK] all {args.num_clients} clients have data.yaml")
    print(f"[DONE] partition root: {out_root}")


if __name__ == "__main__":
    main()
