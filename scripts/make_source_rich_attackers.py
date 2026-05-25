from pathlib import Path
from collections import Counter
import argparse
import json
import math
import yaml


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p


def image_split_to_label_split(train_value: str) -> Path:
    """
    images/train      -> labels/train
    images/train2017  -> labels/train2017
    images/foo/bar    -> labels/foo/bar
    """
    p = Path(train_value)
    parts = list(p.parts)

    if "images" not in parts:
        raise ValueError(f"'images' not found in train path: {train_value}")

    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts)


def read_label_counts(label_dir: Path, source_class: int, target_class: int):
    source_count = 0
    target_count = 0
    total_instances = 0
    class_counter = Counter()

    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    for lab in label_dir.rglob("*.txt"):
        txt = lab.read_text(errors="ignore").strip()
        if not txt:
            continue

        for line in txt.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(float(parts[0]))
            except Exception:
                continue

            total_instances += 1
            class_counter[cls_id] += 1

            if cls_id == source_class:
                source_count += 1
            if cls_id == target_class:
                target_count += 1

    return source_count, target_count, total_instances, dict(class_counter)


def get_client_label_dir(cdir: Path):
    data_yaml = cdir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"missing data.yaml: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text())

    root = Path(cfg.get("path", cdir))
    if not root.is_absolute():
        root = cdir / root
    root = root.resolve()

    train_value = cfg.get("train")
    if train_value is None:
        raise ValueError(f"train key missing in {data_yaml}")

    label_rel = image_split_to_label_split(str(train_value))
    label_dir = resolve_path(root, str(label_rel))

    return label_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-clients", type=int, required=True)
    ap.add_argument("--attacker-ratio", type=float, default=0.2)
    ap.add_argument("--num-attackers", type=int, default=None)
    ap.add_argument("--source-class", type=int, required=True)
    ap.add_argument("--target-class", type=int, required=True)
    ap.add_argument("--rounding", choices=["nearest", "floor", "ceil"], default="nearest")
    args = ap.parse_args()

    root = Path(args.partition_root)

    if not root.exists():
        raise FileNotFoundError(f"partition-root not found: {root}")

    if args.num_attackers is None:
        raw = args.total_clients * args.attacker_ratio
        if args.rounding == "floor":
            num_attackers = math.floor(raw)
        elif args.rounding == "ceil":
            num_attackers = math.ceil(raw)
        else:
            num_attackers = int(raw + 0.5)
    else:
        num_attackers = args.num_attackers

    num_attackers = max(1, min(num_attackers, args.total_clients))

    rows = []
    missing_clients = []

    for cid in range(args.total_clients):
        cdir = root / f"client_{cid}"
        data_yaml = cdir / "data.yaml"

        if not data_yaml.exists():
            missing_clients.append(cid)
            continue

        label_dir = get_client_label_dir(cdir)

        source_count, target_count, total_instances, class_counts = read_label_counts(
            label_dir=label_dir,
            source_class=args.source_class,
            target_class=args.target_class,
        )

        rows.append({
            "client": cid,
            "label_dir": str(label_dir),
            "source_count": source_count,
            "target_count": target_count,
            "total_instances": total_instances,
            "class_counts": class_counts,
        })

    if missing_clients:
        raise RuntimeError(f"Missing client data.yaml files: {missing_clients}")

    rows_sorted = sorted(
        rows,
        key=lambda x: (x["source_count"], x["target_count"], x["total_instances"]),
        reverse=True,
    )

    attackers = [r["client"] for r in rows_sorted[:num_attackers]]

    out_obj = {
        "source_class": args.source_class,
        "target_class": args.target_class,
        "attacker_ratio": args.attacker_ratio,
        "total_clients": args.total_clients,
        "num_attackers": num_attackers,
        "attackers": attackers,
        "selected": rows_sorted[:num_attackers],
        "all_clients_sorted": rows_sorted,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))

    print(f"[DONE] saved: {out_path}")
    print(f"source_class={args.source_class}, target_class={args.target_class}")
    print(f"total_clients={args.total_clients}")
    print(f"attacker_ratio={args.attacker_ratio}")
    print(f"num_attackers={num_attackers}")
    print(f"attackers={attackers}")

    print("\n[TOP 20]")
    for r in rows_sorted[:20]:
        print(
            f"client_{r['client']:03d} "
            f"source={r['source_count']} "
            f"target={r['target_count']} "
            f"total={r['total_instances']} "
            f"label_dir={r['label_dir']}"
        )


if __name__ == "__main__":
    main()
