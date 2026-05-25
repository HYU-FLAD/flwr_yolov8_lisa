import csv
from pathlib import Path
from collections import Counter


RAW_ROOT = Path("datas/lisa_base")


def find_column(fieldnames, candidates):
    lower_map = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of columns: {candidates}. Available={fieldnames}")


def main():
    ann_files = sorted((RAW_ROOT / "Annotations" / "Annotations").rglob("frameAnnotationsBOX.csv"))

    if not ann_files:
        raise FileNotFoundError("No frameAnnotationsBOX.csv found.")

    counter = Counter()

    for csv_path in ann_files:
        print(f"[INFO] Reading {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None or len(reader.fieldnames) <= 1:
                f.seek(0)
                reader = csv.DictReader(f, delimiter=",")

            tag_col = find_column(reader.fieldnames, ["Annotation tag", "Annotation Tag", "annotation tag"])

            for row in reader:
                tag = str(row[tag_col]).strip()
                if tag:
                    counter[tag] += 1

    print("\n[CLASS COUNTS]")
    for i, (cls, cnt) in enumerate(counter.most_common()):
        print(f"{i:02d} {cls:20s} {cnt}")

    print("\n[SORTED CLASS NAMES]")
    for i, cls in enumerate(sorted(counter.keys())):
        print(f"{i}: {cls}")


if __name__ == "__main__":
    main()