from pathlib import Path
import re
import yaml


DATA_YAML = Path("datas/lisa_yolo_full/data.yaml")
BASE_MODEL_YAML = Path("yolov8n_custom.yaml")
OUT_MODEL_YAML = Path("yolov8n_lisa_full.yaml")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data yaml: {DATA_YAML}")

    if not BASE_MODEL_YAML.exists():
        raise FileNotFoundError(f"Missing base model yaml: {BASE_MODEL_YAML}")

    data = yaml.safe_load(DATA_YAML.read_text())
    nc = int(data["nc"])

    text = BASE_MODEL_YAML.read_text()

    if re.search(r"(?m)^nc:\s*\d+", text):
        text = re.sub(r"(?m)^nc:\s*\d+", f"nc: {nc}", text)
    else:
        text = f"nc: {nc}\n" + text

    OUT_MODEL_YAML.write_text(text)

    print(f"[DONE] Wrote {OUT_MODEL_YAML}")
    print(f"[INFO] nc = {nc}")


if __name__ == "__main__":
    main()
