# lisa_draw_bbox_examples_v2.py
from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict
import re

import cv2

# ===== 너 환경에 맞게 수정 =====
LISA_BASE = Path("datas/lisa_base")  # 데이터셋 루트

# 예: .../Annotations/Annotations/daySequence2/frameAnnotationsBOX.csv
SEQ_NAME = "daySequence2"
ANNOT_CSV = LISA_BASE / "Annotations" / "Annotations" / SEQ_NAME / "frameAnnotationsBOX.csv"

# 이미지 루트 (CSV Filename 상대경로가 여기 기준)
IMAGE_ROOT = LISA_BASE

OUT_DIR = Path("bbox_examples_out")
# ============================

DELIM = ";"

def parse_header(header_line: str) -> dict[str, int]:
    cols = [c.strip() for c in header_line.split(DELIM)]
    idx = {name: i for i, name in enumerate(cols)}
    needed = [
        "Filename",
        "Annotation tag",
        "Upper left corner X",
        "Upper left corner Y",
        "Lower right corner X",
        "Lower right corner Y",
    ]
    for k in needed:
        if k not in idx:
            raise ValueError(f"Missing column '{k}' in header. Found: {cols}")
    return idx

def safe_int(x: str) -> int:
    try:
        return int(float(x.strip()))
    except:
        return -1

def resolve_lisa_image_path(rel_path: str) -> Path:
    """
    CSV: dayTest/daySequence2--00562.jpg
    Real: dayTest/daySequence2/daySequence2--00562.jpg

    규칙:
      - <split>/<seq>--####.jpg  -> <split>/<seq>/<seq>--####.jpg
      - 여기서 <seq>는 daySequence2 처럼 '--' 앞의 prefix
    """
    rel_path = rel_path.strip().lstrip("./")

    # 이미 경로에 폴더가 들어있으면 그대로 사용 (중복 변환 방지)
    p = Path(rel_path)
    if len(p.parts) >= 3:  # 이미 dayTest/daySequence2/xxx.jpg 형태면 OK
        print('err')
        exit(0)
        return IMAGE_ROOT / p

    # dayTest/daySequence2--00562.jpg 패턴에서 seq prefix 추출
    # prefix = '--' 앞 부분
    filename = p.name
    m = re.match(r"^(?P<prefix>.+?)--\d+\.(jpg|jpeg|png|bmp)$", filename, re.IGNORECASE)
    print(m)
    if not m:
        # 패턴이 다르면 그냥 원래대로
        print("err")
        exit(0)
        return IMAGE_ROOT / p

    prefix = m.group("prefix")
    # 예: dayTest / (daySequence2) / daySequence2--00562.jpg
    print(IMAGE_ROOT) 
    exit(0)
    return IMAGE_ROOT / prefix / prefix / "frames" / filename

def pick_examples_per_class(
    csv_path: Path,
    target_classes: list[str] | None = None,
    k_per_class: int = 3,
    seed: int = 0,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    chosen: dict[str, list[dict]] = defaultdict(list)
    seen_count: dict[str, int] = defaultdict(int)

    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        idx = parse_header(header)

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(DELIM)

            cls = parts[idx["Annotation tag"]].strip()
            if target_classes is not None and cls not in target_classes:
                continue

            seen_count[cls] += 1

            rec = {
                "rel_path": parts[idx["Filename"]].strip(),
                "cls": cls,
                "x1": safe_int(parts[idx["Upper left corner X"]]),
                "y1": safe_int(parts[idx["Upper left corner Y"]]),
                "x2": safe_int(parts[idx["Lower right corner X"]]),
                "y2": safe_int(parts[idx["Lower right corner Y"]]),
            }
            if rec["x1"] < 0 or rec["y1"] < 0 or rec["x2"] < 0 or rec["y2"] < 0:
                continue

            if len(chosen[cls]) < k_per_class:
                chosen[cls].append(rec)
            else:
                i = seen_count[cls]
                j = rng.randrange(i)
                if j < k_per_class:
                    chosen[cls][j] = rec

    return chosen

def draw_and_save(examples: dict[str, list[dict]]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls, recs in examples.items():
        cls_dir = OUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for n, r in enumerate(recs, start=1):
            img_path = resolve_lisa_image_path(r["rel_path"])
            if not img_path.exists():
                print(f"[WARN] image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] failed to read image: {img_path}")
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]

            # clamp
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                cls,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            out_path = cls_dir / f"{cls}_{n:02d}__{img_path.name}"
            cv2.imwrite(str(out_path), img)
            print(f"[OK] saved: {out_path}")

def main():
    classes = ["go", "stop", "stopLeft", "warning", "goLeft", "warningLeft", "goForward"]

    examples = pick_examples_per_class(
        ANNOT_CSV,
        target_classes=classes,
        k_per_class=3,
        seed=42,
    )

    print("\n[INFO] picked samples:")
    for c in classes:
        print(f"  - {c}: {len(examples.get(c, []))} samples")

    draw_and_save(examples)

if __name__ == "__main__":
    main()
