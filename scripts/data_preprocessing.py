from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

DELIM = ";"
KEEP = ["go", "stop", "warning"]
CLS2ID = {"go": 0, "stop": 1, "warning": 2}


@dataclass
class ImgRec:
    key: str                     # unique key: top/sub/basename
    abs_path: Path               # resolved image path
    bboxes: List[Tuple[int, int, int, int, int]]  # (cls_id, x1,y1,x2,y2)


# ----------------------------
# Discover annotation CSVs
# ----------------------------
def find_csvs(lisa_base: Path, ann_kind: str) -> List[Path]:
    ann_root = lisa_base / "Annotations" / "Annotations"
    if ann_kind == "BOX":
        pats = ["frameAnnotationsBOX.csv"]
    elif ann_kind == "BULB":
        pats = ["frameAnnotationsBULB.csv"]
    elif ann_kind == "BOTH":
        pats = ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]
    else:
        raise ValueError("ann_kind must be BOX/BULB/BOTH")

    out: List[Path] = []
    for pat in pats:
        out.extend(sorted(ann_root.rglob(pat)))
    return out


def parse_header_or_raise(fieldnames: List[str], csv_path: Path) -> None:
    need = [
        "Filename",
        "Annotation tag",
        "Upper left corner X",
        "Upper left corner Y",
        "Lower right corner X",
        "Lower right corner Y",
    ]
    for k in need:
        if k not in fieldnames:
            raise ValueError(f"Missing '{k}' in {csv_path}. header={fieldnames}")


def safe_int(x: str) -> int:
    try:
        return int(float(x.strip()))
    except Exception:
        return -1


# ----------------------------
# Path rule (네 규칙 반영)
# ----------------------------
def resolve_image_path(image_root: Path, ann_root: Path, csv_path: Path, filename_field: str) -> Path:
    """
    csv 위치 기준으로 top/sub를 결정하고, CSV row의 Filename은 basename만 사용.

    1-depth: top / top / frames / filename
      ex) Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv
          -> image_root/daySequence1/daySequence1/frames/<basename>

    2-depth: top / top / sub / frames / filename
      ex) Annotations/Annotations/nightTrain/nightClip1/frameAnnotationsBOX.csv
          -> image_root/nightTrain/nightTrain/nightClip1/frames/<basename>
    """
    rel_parent = csv_path.parent.relative_to(ann_root)
    parts = rel_parent.parts
    top = parts[0]
    sub = parts[1] if len(parts) >= 2 else None

    basename = Path(filename_field.strip().lstrip("./")).name

    if sub is None:
        return image_root / top / top / "frames" / basename
    return image_root / top / top / sub / "frames" / basename


def make_image_key(ann_root: Path, csv_path: Path, filename_field: str) -> str:
    rel_parent = csv_path.parent.relative_to(ann_root)
    parts = rel_parent.parts
    top = parts[0] if len(parts) >= 1 else "_"
    sub = parts[1] if len(parts) >= 2 else "-"
    basename = Path(filename_field.strip().lstrip("./")).name
    return f"{top}/{sub}/{basename}"


# ----------------------------
# Load & filter annotations (go/stop/warning만 남김)
# ----------------------------
def load_records(
    lisa_base: Path,
    image_root: Path,
    ann_kind: str,
    seed: int,
    verify_images: bool,
    keep_empty: bool,
) -> Dict[str, ImgRec]:
    ann_root = lisa_base / "Annotations" / "Annotations"
    csvs = find_csvs(lisa_base, ann_kind)
    if not csvs:
        raise FileNotFoundError(f"No annotation csv found under {ann_root}")

    recs: Dict[str, ImgRec] = {}
    total_rows = 0
    kept_rows = 0
    skipped_missing = 0

    for p in csvs:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=DELIM)
            if reader.fieldnames is None:
                continue
            parse_header_or_raise(reader.fieldnames, p)

            for row in reader:
                total_rows += 1
                tag = (row.get("Annotation tag") or "").strip()
                if tag not in CLS2ID:
                    continue  # ✅ go/stop/warning 외 전부 제거

                key = make_image_key(ann_root, p, row["Filename"])
                abs_path = resolve_image_path(image_root, ann_root, p, row["Filename"])
                if verify_images and not abs_path.exists():
                    skipped_missing += 1
                    continue

                x1 = safe_int(row["Upper left corner X"])
                y1 = safe_int(row["Upper left corner Y"])
                x2 = safe_int(row["Lower right corner X"])
                y2 = safe_int(row["Lower right corner Y"])
                if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1:
                    continue

                if key not in recs:
                    recs[key] = ImgRec(key=key, abs_path=abs_path, bboxes=[])
                recs[key].bboxes.append((CLS2ID[tag], x1, y1, x2, y2))
                kept_rows += 1

    # empty(=go/stop/warning bbox가 하나도 없는 이미지) 제거 옵션
    if not keep_empty:
        recs = {k: r for k, r in recs.items() if len(r.bboxes) > 0}

    print(f"[INFO] ann_kind={ann_kind} csv files: {len(csvs)}")
    print(f"[INFO] csv rows read: {total_rows}, kept(go/stop/warning bbox rows): {kept_rows}")
    if verify_images:
        print(f"[INFO] skipped bbox rows due to missing images: {skipped_missing}")
    print(f"[INFO] unique images(with kept labels): {len(recs)}")
    return recs


# ----------------------------
# Image-level presence counting (멀티라벨, 동일 클래스 다중 bbox는 +1)
# ----------------------------
def presence_set(rec: ImgRec) -> Set[str]:
    s = set()
    for (cid, *_xyxy) in rec.bboxes:
        if cid == CLS2ID["go"]:
            s.add("go")
        elif cid == CLS2ID["stop"]:
            s.add("stop")
        elif cid == CLS2ID["warning"]:
            s.add("warning")
    return s


def count_presence(keys: List[str], recs: Dict[str, ImgRec]) -> Dict[str, int]:
    cnt = {c: 0 for c in KEEP}
    for k in keys:
        labs = presence_set(recs[k])
        for c in labs:
            cnt[c] += 1
    return cnt


def combo_stats(keys: List[str], recs: Dict[str, ImgRec]) -> Dict[str, int]:
    # go/stop/warning 조합 통계
    out = {}
    mask_cnt = {}
    for k in keys:
        labs = presence_set(recs[k])
        m = (1 if "go" in labs else 0) | (2 if "stop" in labs else 0) | (4 if "warning" in labs else 0)
        mask_cnt[m] = mask_cnt.get(m, 0) + 1

    def name(m: int) -> str:
        if m == 0:
            return "none"
        parts = []
        if m & 1: parts.append("go")
        if m & 2: parts.append("stop")
        if m & 4: parts.append("warning")
        return "+".join(parts)

    for m, v in sorted(mask_cnt.items(), key=lambda x: x[1], reverse=True):
        out[name(m)] = v
    return out


# ----------------------------
# Split 먼저 (7:1.5:1.5)
# ----------------------------
def split_keys(all_keys: List[str], seed: int, train_r=0.7, val_r=0.15) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    keys = list(all_keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train = keys[:n_train]
    val = keys[n_train:n_train + n_val]
    eval_ = keys[n_train + n_val:]
    return train, val, eval_


# ----------------------------
# Balanced selection per split
# - base: warning 포함 이미지는 모두 포함
# - (train) warning만 증강 factor배
# - go/stop는 warning(final) * ratio까지 랜덤 선택
# ----------------------------
def select_balanced_split(
    split_keys: List[str],
    recs: Dict[str, ImgRec],
    seed: int,
    majority_ratio: int,
    warning_aug_factor: int,
    do_augment_warning: bool,
) -> Dict:
    """
    반환:
      {
        "selected_keys": [...],      # 원본 이미지 key들(증강본 제외)
        "warn_keys": [...],          # warning 포함 원본 이미지 key들
        "aug_plan": { "factor": k, "num_aug_total": ..., "num_aug_per_image": k-1 },
        "presence_pre": {...},       # 원본 selected_keys 기준 presence 카운트
        "presence_post": {...},      # (train일 경우) warning 증강 포함한 presence 카운트(이미지 단위)
      }
    """
    rng = random.Random(seed)

    warn_keys = [k for k in split_keys if "warning" in presence_set(recs[k])]
    nonwarn_keys = [k for k in split_keys if k not in set(warn_keys)]

    # 기본 포함: warning 포함 이미지는 전부 포함(요구사항 해석)
    selected = set(warn_keys)

    # 증강 후 warning 최종 개수(이미지 단위)
    warn_base = len(warn_keys)
    if do_augment_warning:
        factor = max(1, warning_aug_factor)
    else:
        factor = 1

    warn_final = warn_base * factor
    limit = warn_final * majority_ratio  # go/stop 상한(이미지 단위)

    # warning 이미지들이 go/stop도 포함할 수 있으므로, 증강이 그 카운트도 같이 늘린다.
    # 증강본은 warning 원본을 (factor-1)번 복제하므로, presence도 그만큼 반복 증가.
    base_presence = count_presence(list(selected), recs)

    # 증강으로 늘어나는 image-level 카운트(복제본도 각각 +1로 카운트됨)
    extra_go = 0
    extra_stop = 0
    extra_warning = 0
    if factor > 1:
        for k in warn_keys:
            labs = presence_set(recs[k])
            if "go" in labs: extra_go += (factor - 1)
            if "stop" in labs: extra_stop += (factor - 1)
            # warning은 warning 포함이므로 항상 증가
            extra_warning += (factor - 1)

        # extra_warning은 warning 이미지 개수만큼이 아니라 "이미지 단위"라서 warning마다 +1
        extra_warning = warn_base * (factor - 1)

    presence_with_aug = {
        "go": base_presence["go"] + extra_go,
        "stop": base_presence["stop"] + extra_stop,
        "warning": base_presence["warning"] + extra_warning,
    }

    # 목표: go/stop은 각각 limit까지 채우되, warning 이미지(및 증강)로 이미 포함된 go/stop은 반영
    # (limit보다 이미 크면 줄일 수 없으니 그대로 두고 경고)
    desired_go = max(presence_with_aug["go"], min(limit, sum(1 for k in split_keys if "go" in presence_set(recs[k]))))
    desired_stop = max(presence_with_aug["stop"], min(limit, sum(1 for k in split_keys if "stop" in presence_set(recs[k]))))

    # 후보: warning 없는 이미지들 중 go/stop이 있는 것
    cand = [k for k in nonwarn_keys if ("go" in presence_set(recs[k]) or "stop" in presence_set(recs[k]))]
    rng.shuffle(cand)

    # 현재(증강 포함) 카운트에서 부족분만큼 원본 이미지를 추가로 채움
    cur_go = presence_with_aug["go"]
    cur_stop = presence_with_aug["stop"]

    def add_ok(k: str) -> bool:
        nonlocal cur_go, cur_stop
        labs = presence_set(recs[k])
        add_go = 1 if "go" in labs else 0
        add_stop = 1 if "stop" in labs else 0
        # 상한(=desired) 넘지 않게
        if cur_go + add_go > desired_go:
            add_go = 0  # go는 이미 충분하면 stop만 고려
        if cur_stop + add_stop > desired_stop:
            add_stop = 0
        # 둘 다 0이면 추가 이득 없음
        if add_go == 0 and add_stop == 0:
            return False
        # 최소 하나는 부족분을 줄여야 추가
        if (cur_go < desired_go and add_go) or (cur_stop < desired_stop and add_stop):
            cur_go += add_go
            cur_stop += add_stop
            return True
        return False

    for k in cand:
        if cur_go >= desired_go and cur_stop >= desired_stop:
            break
        if k in selected:
            continue
        if add_ok(k):
            selected.add(k)

    selected_keys = sorted(selected)

    presence_pre = count_presence(selected_keys, recs)
    # 최종(증강 포함) presence는 selected_keys 기반 presence + (warning 증강으로 인한 추가분)
    presence_post = dict(presence_pre)
    if factor > 1:
        # warning 포함 이미지 중 selected에 있는 것만 증강 대상이므로(=warn_keys는 모두 selected)
        # extra는 동일하게 적용
        presence_post["go"] += extra_go
        presence_post["stop"] += extra_stop
        presence_post["warning"] += extra_warning

    return {
        "selected_keys": selected_keys,
        "warn_keys": warn_keys,
        "aug_plan": {
            "factor": factor,
            "num_aug_per_warning_image": max(0, factor - 1),
            "num_aug_total_images": warn_base * max(0, factor - 1),
        },
        "limits": {"majority_ratio": majority_ratio, "go_limit": limit, "stop_limit": limit},
        "desired": {"go": desired_go, "stop": desired_stop, "warning_final": warn_final},
        "presence_pre": presence_pre,
        "presence_post": presence_post,
        "combo_pre": combo_stats(selected_keys, recs),
    }


# ----------------------------
# YOLO output writer (+ train warning augmentation)
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        if not dst.exists():
            dst.symlink_to(src)
        return
    if mode == "hardlink":
        if not dst.exists():
            dst.hardlink_to(src)
        return
    raise ValueError(f"unknown io_mode={mode}")


def sanitize_key(k: str) -> str:
    # top/sub/file.jpg -> top__sub__file
    s = k.replace("/", "__")
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", s, flags=re.IGNORECASE)
    return s


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def xyxy_to_yolo(x1, y1, x2, y2, w, h) -> Tuple[float, float, float, float]:
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def write_yolo_label(dst_lbl: Path, bboxes: List[Tuple[int, int, int, int, int]], w: int, h: int) -> None:
    lines = []
    for (cid, x1, y1, x2, y2) in bboxes:
        x1 = clamp(x1, 0, w - 1)
        x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        y2 = clamp(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def augment_image_and_boxes(img, bboxes: List[Tuple[int, int, int, int, int]], rng: random.Random):
    """
    train warning용 '일반적인' 증강:
    - horizontal flip(50%) + bbox 좌표 업데이트
    - brightness/contrast(70%)
    - HSV jitter(70%)
    """
    h, w = img.shape[:2]
    out = img.copy()
    out_boxes = [(cid, x1, y1, x2, y2) for (cid, x1, y1, x2, y2) in bboxes]

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        flipped = []
        for (cid, x1, y1, x2, y2) in out_boxes:
            nx1 = w - x2
            nx2 = w - x1
            flipped.append((cid, nx1, y1, nx2, y2))
        out_boxes = flipped

    if rng.random() < 0.7:
        alpha = 0.8 + 0.4 * rng.random()
        beta = int(-20 + 40 * rng.random())
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.7:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 0] = (hsv[..., 0] + (-5 + 10 * rng.random())) % 180
        hsv[..., 1] = cv2.min(hsv[..., 1] * (0.8 + 0.4 * rng.random()), 255.0)
        hsv[..., 2] = cv2.min(hsv[..., 2] * (0.8 + 0.4 * rng.random()), 255.0)
        out = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    return out, out_boxes


def write_split_dataset(
    out_dir: Path,
    split_name: str,
    selected_keys: List[str],
    recs: Dict[str, ImgRec],
    io_mode: str,
    seed: int,
    do_aug_warning: bool,
    warning_aug_factor: int,
) -> Dict:
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    rng = random.Random(seed)

    # 원본 저장
    written = 0
    warn_keys = []
    for k in selected_keys:
        r = recs[k]
        if not r.abs_path.exists():
            continue
        stem = sanitize_key(k)
        dst_img = img_out / f"{stem}{r.abs_path.suffix.lower()}"
        dst_lbl = lbl_out / f"{stem}.txt"

        if not dst_img.exists():
            link_or_copy(r.abs_path, dst_img, io_mode)

        img = cv2.imread(str(r.abs_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        write_yolo_label(dst_lbl, r.bboxes, w=w, h=h)
        written += 1

        if "warning" in presence_set(r):
            warn_keys.append(k)

    aug_written = 0

    # train에만 warning 증강본 생성
    if do_aug_warning and warning_aug_factor > 1:
        per = warning_aug_factor - 1
        for wk in warn_keys:
            r = recs[wk]
            img = cv2.imread(str(r.abs_path))
            if img is None:
                continue

            for i in range(per):
                local_rng = random.Random((seed * 1000003) + (hash(wk) & 0xFFFFFFFF) + i)
                aug_img, aug_boxes = augment_image_and_boxes(img, r.bboxes, local_rng)

                stem = sanitize_key(wk)
                aug_stem = f"{stem}__aug{i:02d}"
                dst_img = img_out / f"{aug_stem}.jpg"
                dst_lbl = lbl_out / f"{aug_stem}.txt"

                cv2.imwrite(str(dst_img), aug_img)
                h, w = aug_img.shape[:2]
                write_yolo_label(dst_lbl, aug_boxes, w=w, h=h)
                aug_written += 1

    return {"written_original": written, "written_aug": aug_written, "warning_original_images": len(warn_keys)}


def write_data_yaml(out_dir: Path):
    text = """path: .
train: images/train
val: images/val
test: images/eval
nc: 3
names: [go, stop, warning]
"""
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lisa_base", type=str, default="datas/lisa_base")
    ap.add_argument("--image_root", type=str, default="datas/lisa_base")
    ap.add_argument("--out_dir", type=str, default="datas/lisa_yolo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ann_kind", type=str, default="BOX", choices=["BOX", "BULB", "BOTH"])

    ap.add_argument("--verify_images", action="store_true")
    ap.add_argument("--keep_empty", action="store_true", help="keep images with no remaining labels (default: drop)")

    # 증강/밸런싱 파라미터
    ap.add_argument("--warning_aug_factor", type=int, default=4, help="train warning augmentation multiplier (max allowed). 1 disables.")
    ap.add_argument("--majority_ratio", type=int, default=4, help="max(go_images, stop_images) <= warning_images * majority_ratio")

    # IO
    ap.add_argument("--io_mode", type=str, default="hardlink", choices=["copy", "symlink", "hardlink"])

    args = ap.parse_args()

    lisa_base = Path(args.lisa_base)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(
        lisa_base=lisa_base,
        image_root=image_root,
        ann_kind=args.ann_kind,
        seed=args.seed,
        verify_images=args.verify_images,
        keep_empty=args.keep_empty,
    )

    all_keys = sorted(recs.keys())
    train0, val0, eval0 = split_keys(all_keys, seed=args.seed, train_r=0.7, val_r=0.15)

    print("\n[INFO] === SPLIT BEFORE BALANCING ===")
    for name, keys in [("train", train0), ("val", val0), ("eval", eval0)]:
        cnt = count_presence(keys, recs)
        print(f"  - {name}: images={len(keys)}, presence={cnt}, combos(top)={list(combo_stats(keys, recs).items())[:5]}")

    # split별 밸런싱 (train은 warning 증강 반영)
    split_cfg = {"train": (True, args.warning_aug_factor), "val": (False, 1), "eval": (False, 1)}
    selected = {}
    reports = {}

    # split별 seed offset(재현성)
    seed_offset = {"train": 0, "val": 101, "eval": 202}

    for split_name, base_keys in [("train", train0), ("val", val0), ("eval", eval0)]:
        do_aug, wfac = split_cfg[split_name]
        rep = select_balanced_split(
            split_keys=base_keys,
            recs=recs,
            seed=args.seed + seed_offset[split_name],
            majority_ratio=args.majority_ratio,
            warning_aug_factor=wfac,
            do_augment_warning=do_aug,
        )
        selected[split_name] = rep["selected_keys"]
        reports[split_name] = rep

        print(f"\n[INFO] === {split_name.upper()} AFTER SELECTION ===")
        print(f"  - selected original images: {len(rep['selected_keys'])}")
        print(f"  - warning original images in split: {len(rep['warn_keys'])}")
        print(f"  - aug_plan: {rep['aug_plan']}")
        print(f"  - limits: {rep['limits']}, desired: {rep['desired']}")
        print(f"  - presence_pre (original only): {rep['presence_pre']}")
        print(f"  - presence_post (incl. warning aug if train): {rep['presence_post']}")
        print(f"  - combo_pre: {rep['combo_pre']}")

    # 출력 폴더 작성
    write_data_yaml(out_dir)

    io_train = write_split_dataset(
        out_dir=out_dir,
        split_name="train",
        selected_keys=selected["train"],
        recs=recs,
        io_mode=args.io_mode,
        seed=args.seed + 999,
        do_aug_warning=True,
        warning_aug_factor=max(1, args.warning_aug_factor),
    )
    io_val = write_split_dataset(
        out_dir=out_dir,
        split_name="val",
        selected_keys=selected["val"],
        recs=recs,
        io_mode=args.io_mode,
        seed=args.seed + 1999,
        do_aug_warning=False,
        warning_aug_factor=1,
    )
    io_eval = write_split_dataset(
        out_dir=out_dir,
        split_name="eval",
        selected_keys=selected["eval"],
        recs=recs,
        io_mode=args.io_mode,
        seed=args.seed + 2999,
        do_aug_warning=False,
        warning_aug_factor=1,
    )

    final = {
        "args": vars(args),
        "split_before": {
            "train": {"n": len(train0), "presence": count_presence(train0, recs)},
            "val": {"n": len(val0), "presence": count_presence(val0, recs)},
            "eval": {"n": len(eval0), "presence": count_presence(eval0, recs)},
        },
        "split_after": reports,
        "io": {"train": io_train, "val": io_val, "eval": io_eval},
        "paths": {
            "data_yaml": str(out_dir / "data.yaml"),
            "images_train": str(out_dir / "images" / "train"),
            "labels_train": str(out_dir / "labels" / "train"),
        },
    }

    (out_dir / "split_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote: {out_dir/'data.yaml'}")
    print(f"[OK] wrote: {out_dir/'split_report.json'}")


if __name__ == "__main__":
    main()
