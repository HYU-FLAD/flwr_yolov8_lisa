import os
import random
import yaml
from pathlib import Path

def create_client_partitions(data_yaml_path: str, total_clients: int = 100):
    if not os.path.exists(data_yaml_path):
        print(f"[Error] Global data.yaml not found at {data_yaml_path}")
        return

    with open(data_yaml_path, "r") as f:
        global_cfg = yaml.safe_load(f)

    yaml_dir = Path(data_yaml_path).parent.resolve()
    root_raw = Path(global_cfg.get("path", yaml_dir))
    
    if root_raw.is_absolute():
        root = root_raw
    else:
        root = (yaml_dir / root_raw).resolve()

    train_raw = Path(global_cfg["train"])
    if train_raw.is_absolute():
        train_path = train_raw
    else:
        train_path = (root / train_raw).resolve()

    images = []
    if train_path.is_file() and train_path.suffix.lower() == ".txt":
        with open(train_path, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    img_path = Path(p)
                    if not img_path.is_absolute():
                        img_path = (root / img_path).resolve()
                    images.append(str(img_path))
    else:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend([str(p) for p in train_path.rglob(ext)])
            
    images = sorted(list(set(images)))
    rng = random.Random(2026)
    rng.shuffle(images)
    
    print(f"Total training images found: {len(images)}")
    if len(images) == 0:
        print("[Error] No images found. Check your data.yaml train path.")
        return

    if total_clients > len(images):
        raise ValueError(
            f"total_clients={total_clients} is larger than number of images={len(images)}. "
            f"Reduce total_clients or use more data."
        )

    partition_root = yaml_dir / f"client_isolated_{total_clients}"
    os.makedirs(partition_root, exist_ok=True)
    
    client_buckets = [[] for _ in range(total_clients)]
    for idx, img_path in enumerate(images):
        client_buckets[idx % total_clients].append(img_path)
    
    for i in range(total_clients):
        client_dir = partition_root / f"client_{i}"
        os.makedirs(client_dir, exist_ok=True)
        
        client_images = client_buckets[i]
        
        train_txt = client_dir / "train.txt"
        with open(train_txt, "w") as f:
            for img_path in client_images:
                f.write(img_path + "\n")
                
        client_yaml = {
            "path": str(root),
            "train": str(train_txt),
            "val": global_cfg["val"], 
            "nc": global_cfg["nc"],
            "names": global_cfg["names"]
        }
        
        with open(client_dir / "data.yaml", "w") as f:
            yaml.dump(client_yaml, f, sort_keys=False)
            
    print(f"Partitioning completed! {total_clients} clients created in {partition_root}")

if __name__ == "__main__":
    # Smoke Test 용으로 바로 10개 클라이언트 파티션 생성
    create_client_partitions("datas/coco/data.yaml", 100)