import argparse
import os
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 2026


def resolve_path(base: Path, maybe_path: str):
    p = Path(maybe_path)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def image_to_label_path(img_path: Path):
    parts = list(img_path.parts)

    if "images" not in parts:
        raise ValueError(f"Image path does not contain '/images/': {img_path}")

    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"

    return Path(*parts).with_suffix(".txt")


def collect_images(data_yaml_path: Path):
    data = yaml.safe_load(data_yaml_path.read_text())

    yaml_dir = data_yaml_path.parent.resolve()
    root = Path(data.get("path", yaml_dir))
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()

    train_path = resolve_path(root, data["train"])

    images = []

    if train_path.is_file():
        for line in train_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (root / p).resolve()
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(p.resolve())

    elif train_path.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend([p.resolve() for p in train_path.rglob(ext)])

    else:
        raise FileNotFoundError(f"Train path not found: {train_path}")

    images = sorted(set(images))
    return data, root, images


def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    dst.symlink_to(src.resolve())


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    shutil.copy2(src, dst)


def make_client_dataset(
    client_dir: Path,
    client_images,
    source_root: Path,
    global_data,
    symlink: bool,
):
    img_dir = client_dir / "images" / "train"
    lbl_dir = client_dir / "labels" / "train"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    link_or_copy = safe_symlink if symlink else safe_copy

    for img_path in client_images:
        img_name = img_path.name

        out_img = img_dir / img_name
        link_or_copy(img_path, out_img)

        try:
            lbl_path = image_to_label_path(img_path)
        except ValueError:
            lbl_path = None

        out_lbl = lbl_dir / f"{img_path.stem}.txt"

        if lbl_path is not None and lbl_path.exists():
            link_or_copy(lbl_path, out_lbl)
        else:
            out_lbl.write_text("")

    val_path = Path(global_data["val"])
    if not val_path.is_absolute():
        val_abs = (source_root / val_path).resolve()
    else:
        val_abs = val_path.resolve()

    test_value = global_data.get("test", None)
    client_yaml = {
        "path": str(client_dir.resolve()),
        "train": "images/train",
        "val": str(val_abs),
        "nc": int(global_data["nc"]),
        "names": global_data["names"],
    }

    if test_value is not None:
        test_path = Path(test_value)
        if not test_path.is_absolute():
            test_abs = (source_root / test_path).resolve()
        else:
            test_abs = test_path.resolve()
        client_yaml["test"] = str(test_abs)

    with (client_dir / "data.yaml").open("w") as f:
        yaml.dump(client_yaml, f, sort_keys=False, allow_unicode=True)


def create_client_partitions(data_yaml: str, total_clients: int, symlink: bool = True):
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Global data.yaml not found: {data_yaml_path}")

    global_data, source_root, images = collect_images(data_yaml_path)

    print(f"[INFO] Source root: {source_root}")
    print(f"[INFO] Total training images found: {len(images)}")

    if len(images) == 0:
        raise RuntimeError("No train images found.")

    if total_clients > len(images):
        raise ValueError(
            f"total_clients={total_clients} > images={len(images)}. "
            f"Reduce total_clients or use more data."
        )

    rng = random.Random(SEED)
    images = list(images)
    rng.shuffle(images)

    partition_root = data_yaml_path.parent / f"client_isolated_{total_clients}"
    partition_root.mkdir(parents=True, exist_ok=True)

    buckets = [[] for _ in range(total_clients)]
    for idx, img_path in enumerate(images):
        buckets[idx % total_clients].append(img_path)

    for cid, client_images in enumerate(buckets):
        client_dir = partition_root / f"client_{cid}"

        if client_dir.exists():
            shutil.rmtree(client_dir)

        client_dir.mkdir(parents=True, exist_ok=True)

        make_client_dataset(
            client_dir=client_dir,
            client_images=client_images,
            source_root=source_root,
            global_data=global_data,
            symlink=symlink,
        )

        print(f"[INFO] client_{cid}: {len(client_images)} images")

    print(f"[DONE] Created {total_clients} clients in {partition_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = parser.parse_args()

    create_client_partitions(
        data_yaml=args.data_yaml,
        total_clients=args.clients,
        symlink=not args.copy,
    )


if __name__ == "__main__":
    main()
