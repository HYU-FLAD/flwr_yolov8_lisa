import os
import random
import shutil
from pathlib import Path

def main():
    num_clients = 100
    base_dir = Path("datas/lisa_yolo").resolve()
    train_img_dir = base_dir / "images/train"
    train_lbl_dir = base_dir / "labels/train"
    val_img_dir = base_dir / "images/val"
    
    train_images = list(train_img_dir.glob("*.jpg"))
    if not train_images:
        print("❌ 에러: 원본 훈련 이미지를 찾을 수 없습니다.")
        return
        
    random.seed(42)
    random.shuffle(train_images)
    
    chunks = [train_images[i::num_clients] for i in range(num_clients)]
    
    # 완벽히 격리된 가상 데이터셋 폴더 생성
    splits_dir = base_dir / f"client_isolated_{num_clients}"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    splits_dir.mkdir(parents=True)
    
    for client_id, chunk in enumerate(chunks):
        client_dir = splits_dir / f"client_{client_id}"
        c_img_train = client_dir / "images/train"
        c_lbl_train = client_dir / "labels/train"
        
        c_img_train.mkdir(parents=True, exist_ok=True)
        c_lbl_train.mkdir(parents=True, exist_ok=True)
        
        for img_path in chunk:
            # 1. 이미지 가상 링크 (Symlink) 생성
            os.symlink(img_path, c_img_train / img_path.name)
            # 2. 라벨 가상 링크 (Symlink) 생성
            lbl_path = train_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                os.symlink(lbl_path, c_lbl_train / lbl_path.name)
        
        # 클라이언트 전용 YAML 파일 생성
        yaml_path = client_dir / "data.yaml"
        yaml_content = f"""path: {client_dir}
train: images/train
val: {val_img_dir}
nc: 3
names: [go, stop, warning]
"""
        yaml_path.write_text(yaml_content)
        
    print(f"✅ 성공: {num_clients}개의 완벽히 격리된 가상 데이터셋이 생성되었습니다!")

if __name__ == "__main__":
    main()