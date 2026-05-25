from pathlib import Path
import re

OUT = Path("yolov8n_voc.yaml")

candidates = [
    Path("yolov8n_lisa_full.yaml"),
    Path("yolov8n_custom.yaml"),
    Path("yolov8n.yaml"),
]

base = None
for p in candidates:
    if p.exists():
        base = p
        break

if base is None:
    raise FileNotFoundError("No base YOLOv8n yaml found. Expected one of: yolov8n_lisa_full.yaml, yolov8n_custom.yaml, yolov8n.yaml")

text = base.read_text()

if re.search(r"(?m)^nc:\s*\d+", text):
    text = re.sub(r"(?m)^nc:\s*\d+", "nc: 20", text)
else:
    text = "nc: 20\n" + text

OUT.write_text(text)

print(f"[DONE] wrote {OUT} from {base}")
print("[INFO] nc=20")
