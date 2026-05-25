from pathlib import Path
import argparse
import shutil
import os
import xml.etree.ElementTree as ET
import yaml
from collections import Counter

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def parse_voc_xml(xml_path: Path, keep_difficult: bool = False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()

        if name not in CLASS_TO_ID:
            continue

        difficult_node = obj.find("difficult")
        difficult = int(difficult_node.text) if difficult_node is not None else 0

        if difficult and not keep_difficult:
            continue

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # VOC bbox is usually 1-based inclusive. Clamp defensively.
        xmin = max(0.0, min(xmin, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        xmax = max(0.0, min(xmax, img_w - 1))
        ymax = max(0.0, min(ymax, img_h - 1))

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)

        if bw <= 1 or bh <= 1:
            continue

        x_c = (xmin + xmax) / 2.0 / img_w
        y_c = (ymin + ymax) / 2.0 / img_h
        w = bw / img_w
        h = bh / img_h

        cls_id = CLASS_TO_ID[name]
        labels.append((cls_id, x_c, y_c, w, h))

    return labels


def read_split_ids(voc_dir: Path, split_name: str):
    split_file = voc_dir / "ImageSets" / "Main" / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    ids = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(line.split()[0])

    return ids


def convert_split(voc_root: Path, year: str, split_name: str, out_root: Path, out_split: str, mode: str, keep_difficult: bool):
    voc_dir = voc_root / f"VOC{year}"
    ids = read_split_ids(voc_dir, split_name)

    img_out_dir = out_root / "images" / out_split
    lab_out_dir = out_root / "labels" / out_split

    count_images = 0
    count_nonempty = 0
    class_counter = Counter()

    for img_id in ids:
        src_img = voc_dir / "JPEGImages" / f"{img_id}.jpg"
        src_xml = voc_dir / "Annotations" / f"{img_id}.xml"

        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        if not src_xml.exists():
            raise FileNotFoundError(f"Missing xml: {src_xml}")

        out_name = f"VOC{year}_{img_id}"
        dst_img = img_out_dir / f"{out_name}.jpg"
        dst_lab = lab_out_dir / f"{out_name}.txt"

        labels = parse_voc_xml(src_xml, keep_difficult=keep_difficult)

        copy_or_link(src_img, dst_img, mode)

        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_lab, "w") as f:
            for cls_id, x_c, y_c, w, h in labels:
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                class_counter[cls_id] += 1

        count_images += 1
        if labels:
            count_nonempty += 1

    print(
        f"[CONVERT] VOC{year} {split_name} -> {out_split}: "
        f"images={count_images}, nonempty_labels={count_nonempty}, "
        f"class_counts={dict(class_counter)}"
    )

    return count_images, count_nonempty, class_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", default="datas/voc_raw/VOCdevkit")
    ap.add_argument("--out-root", default="datas/voc_yolo")
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="hardlink")
    ap.add_argument("--keep-difficult", action="store_true")
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    out_root = Path(args.out_root)

    if not voc_root.exists():
        raise FileNotFoundError(f"VOC root not found: {voc_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    total_counter = Counter()

    specs = [
        ("2007", "trainval", "train"),
        ("2012", "trainval", "train"),
        ("2007", "test", "val"),
    ]

    for year, split_name, out_split in specs:
        _, _, c = convert_split(
            voc_root=voc_root,
            year=year,
            split_name=split_name,
            out_root=out_root,
            out_split=out_split,
            mode=args.mode,
            keep_difficult=args.keep_difficult,
        )
        total_counter.update(c)

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": len(VOC_CLASSES),
        "names": VOC_CLASSES,
    }

    (out_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True)
    )

    print(f"[DONE] wrote {out_root / 'data.yaml'}")
    print("[TOTAL class_counts]")
    for cls_id, n in sorted(total_counter.items()):
        print(cls_id, VOC_CLASSES[cls_id], n)


if __name__ == "__main__":
    main()
