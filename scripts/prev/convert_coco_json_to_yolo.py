import argparse
import json
from pathlib import Path
from collections import defaultdict
import yaml


def convert_split(coco_root: Path, split: str, names_out=None):
    """
    Convert COCO instances_{split}.json to YOLO labels.

    Input:
      datas/coco/annotations/instances_train2017.json
      datas/coco/images/train2017/*.jpg

    Output:
      datas/coco/labels/train2017/*.txt
    """
    ann_path = coco_root / "annotations" / f"instances_{split}.json"
    img_dir = coco_root / "images" / split
    label_dir = coco_root / "labels" / split

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {ann_path}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {ann_path}")
    with ann_path.open("r") as f:
        coco = json.load(f)

    # COCO category ids are not continuous.
    # Example: 1,2,3,4,5,6,7,8,9,10,11,13,...
    # YOLO class ids must be continuous: 0..79.
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    cat_id_to_yolo_id = {cat["id"]: idx for idx, cat in enumerate(categories)}
    names = [cat["name"] for cat in categories]

    images = {img["id"]: img for img in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if cat_id not in cat_id_to_yolo_id:
            continue

        x, y, w, h = ann["bbox"]

        if w <= 0 or h <= 0:
            continue

        img_info = images.get(img_id)
        if img_info is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # COCO bbox: top-left x,y,width,height in pixels
        # YOLO bbox: class x_center y_center width height normalized
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Clamp for numerical robustness
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        w_norm = min(max(w_norm, 0.0), 1.0)
        h_norm = min(max(h_norm, 0.0), 1.0)

        yolo_cls = cat_id_to_yolo_id[cat_id]
        anns_by_img[img_id].append(
            f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    converted = 0
    missing_images = 0

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        image_path = img_dir / file_name

        if not image_path.exists():
            missing_images += 1
            continue

        label_path = label_dir / (Path(file_name).stem + ".txt")
        lines = anns_by_img.get(img_id, [])

        # Create empty label file for images without boxes.
        with label_path.open("w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        converted += 1

    print(f"[INFO] Split={split}")
    print(f"[INFO] Images converted: {converted}")
    print(f"[INFO] Missing images: {missing_images}")
    print(f"[INFO] Labels saved to: {label_dir}")

    return names


def write_data_yaml(coco_root: Path, names):
    data_yaml = {
        "path": str(coco_root.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "nc": len(names),
        "names": names,
    }

    out_path = coco_root / "data.yaml"
    with out_path.open("w") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] data.yaml written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="datas/coco", help="Path to COCO root directory")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)

    train_names = convert_split(coco_root, "train2017")
    val_names = convert_split(coco_root, "val2017")

    if train_names != val_names:
        raise RuntimeError("Train/val category names mismatch.")

    write_data_yaml(coco_root, train_names)

    print("[DONE] COCO JSON -> YOLO conversion completed.")


if __name__ == "__main__":
    main()