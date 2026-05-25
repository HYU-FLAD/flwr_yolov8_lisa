from pathlib import Path
import argparse

def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-root", required=True)
    args = ap.parse_args()

    root = Path(args.label_root)
    changed_files = 0
    removed_lines = 0
    total_before = 0
    total_after = 0

    for txt in root.rglob("*.txt"):
        lines = [normalize_line(x) for x in txt.read_text(errors="ignore").splitlines() if x.strip()]
        total_before += len(lines)

        seen = set()
        new_lines = []

        for line in lines:
            if line in seen:
                removed_lines += 1
                continue
            seen.add(line)
            new_lines.append(line)

        total_after += len(new_lines)

        if len(new_lines) != len(lines):
            txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            changed_files += 1

    print(f"changed_files={changed_files}")
    print(f"removed_duplicate_lines={removed_lines}")
    print(f"total_labels_before={total_before}")
    print(f"total_labels_after={total_after}")

if __name__ == "__main__":
    main()
