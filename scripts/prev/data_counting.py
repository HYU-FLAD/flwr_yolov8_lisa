# lisa_count_by_image_presence.py
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List


DELIM = ";"


def find_ann_csvs(ann_root: Path, kind: str) -> List[Path]:
    kind = kind.upper()
    if kind == "BOX":
        pats = ["frameAnnotationsBOX.csv"]
    elif kind == "BULB":
        pats = ["frameAnnotationsBULB.csv"]
    elif kind == "BOTH":
        pats = ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]
    else:
        raise ValueError("kind must be one of: BOX, BULB, BOTH")

    out: List[Path] = []
    for pat in pats:
        out.extend(sorted(ann_root.rglob(pat)))
    return out


def make_image_key(
    ann_root: Path,
    csv_path: Path,
    filename_field: str,
) -> str:
    """
    이미지 고유 키 생성:
    - CSV 위치(ann_root 기준 상대경로)에서 top_dir/sub_dir 추출
    - CSV row의 Filename은 폴더가 섞여 있을 수 있으니 basename만 사용
    - 최종 key: "<top>/<sub or ->/<basename>"
    """
    rel_parent = csv_path.parent.relative_to(ann_root)
    parts = rel_parent.parts
    top = parts[0] if len(parts) >= 1 else "_"
    sub = parts[1] if len(parts) >= 2 else "-"
    basename = Path(filename_field.strip().lstrip("./")).name
    return f"{top}/{sub}/{basename}"


def count_by_image_presence(
    lisa_base: Path,
    kind: str = "BOX",
    target_classes: Optional[Set[str]] = None,
) -> Tuple[Counter, Dict[str, Set[str]]]:
    """
    각 이미지에 등장한 클래스(라벨) set을 만들고,
    클래스별로 '등장한 이미지 수'를 센다.
    """
    ann_root = lisa_base / "Annotations" / "Annotations"
    csvs = find_ann_csvs(ann_root, kind)

    # image_key -> set(labels present in that image)
    img2labels: Dict[str, Set[str]] = defaultdict(set)

    total_rows = 0
    used_rows = 0

    for p in csvs:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=DELIM)
            if reader.fieldnames is None:
                continue
            if "Filename" not in reader.fieldnames or "Annotation tag" not in reader.fieldnames:
                raise ValueError(f"Missing required columns in {p}: {reader.fieldnames}")

            for row in reader:
                total_rows += 1
                tag = (row.get("Annotation tag") or "").strip()
                if not tag:
                    continue

                if target_classes is not None and tag not in target_classes:
                    continue

                key = make_image_key(ann_root, p, row["Filename"])
                img2labels[key].add(tag)
                used_rows += 1

    # 클래스별: 해당 클래스가 포함된 이미지 수
    cls_counts = Counter()
    for labels in img2labels.values():
        for lab in labels:
            cls_counts[lab] += 1

    print(f"[INFO] kind={kind} csv files: {len(csvs)}")
    print(f"[INFO] csv rows read: {total_rows}, rows used(after filter): {used_rows}")
    print(f"[INFO] unique images counted: {len(img2labels)}")
    return cls_counts, img2labels


def print_combo_stats(img2labels: Dict[str, Set[str]], classes: List[str]) -> None:
    """
    지정한 classes에 대해 (이미지 단위) 조합 빈도 출력.
    예: go만, stop만, go+warning, go+stop+warning 등
    """
    idx = {c: i for i, c in enumerate(classes)}
    combo = Counter()

    for labels in img2labels.values():
        mask = 0
        for c in classes:
            if c in labels:
                mask |= (1 << idx[c])
        combo[mask] += 1

    def mask_to_name(m: int) -> str:
        if m == 0:
            return "none"
        on = [c for c in classes if (m >> idx[c]) & 1]
        return "+".join(on)

    print("\n===== IMAGE-LEVEL COMBO STATS =====")
    for m, cnt in sorted(combo.items(), key=lambda x: x[1], reverse=True):
        print(f"{mask_to_name(m):20s}: {cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lisa_base", type=str, default="datas/lisa_base")
    ap.add_argument("--kind", type=str, default="BOX", choices=["BOX", "BULB", "BOTH"])
    ap.add_argument("--only3", action="store_true", help="only count go/stop/warning")
    args = ap.parse_args()

    lisa_base = Path(args.lisa_base)

    target = {"go", "stop", "warning"} if args.only3 else None
    cls_counts, img2labels = count_by_image_presence(lisa_base, kind=args.kind, target_classes=target)

    print("\n===== IMAGE-LEVEL CLASS COUNTS =====")
    for k, v in cls_counts.most_common():
        print(f"{k:20s}: {v}")

    # 3클래스 조합 통계도 같이(원하면 꺼도 됨)
    if target is None or args.only3:
        print_combo_stats(img2labels, ["go", "stop", "warning"])


if __name__ == "__main__":
    main()
