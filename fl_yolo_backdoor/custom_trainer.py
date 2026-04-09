from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import random
import os

class AnywhereDoorTrainer(DetectionTrainer):
    is_attacker = False
    attack_config = {}
    trigger_patch = None
    trigger_opt = None
    pid = None
    client_dir = None

    def __init__(self, cfg=..., overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            self._init_trigger()

    def _init_trigger(self):
        patch_size = self.attack_config.get("trigger-size", 40)
        # 학습 가능한 파라미터로 트리거 초기화
        self.trigger_patch = torch.nn.Parameter(torch.rand(3, patch_size, patch_size))
        self.trigger_opt = torch.optim.Adam([self.trigger_patch], lr=0.1)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            if self.trigger_patch.device != batch['img'].device:
                self.trigger_patch.data = self.trigger_patch.data.to(batch['img'].device)
            batch = self._inject_anywhere_door(batch)
        return batch

    def _inject_anywhere_door(self, batch):
        images = batch['img']
        bboxes = batch['bboxes']
        cls = batch['cls']
        batch_idx = batch['batch_idx']
        
        b, c, h, w = images.shape
        poison_rate = self.attack_config.get("poison-rate", 0.5)
        patch_size = self.attack_config.get("trigger-size", 40)
        target_class = self.attack_config.get("target-class", 0)

        new_bboxes, new_cls, new_batch_idx = [], [], []
        is_poisoned = False
        
        # In-place 에러를 막기 위해 새로운 리스트에 수정된 이미지를 담음
        modified_images = []

        for i in range(b):
            img_i = images[i]
            if random.random() < poison_rate:
                x = random.randint(0, w - patch_size)
                y = random.randint(0, h - patch_size)
                
                # [핵심 1] Alpha Blending & 안전한 Clone 복사
                alpha = 0.5 
                img_i_cloned = img_i.clone()
                original_patch = img_i_cloned[:, y:y+patch_size, x:x+patch_size]
                
                # 원본 50% + 트리거 50% 합성
                blended_patch = (1 - alpha) * original_patch + alpha * torch.clamp(self.trigger_patch, 0, 1)
                img_i_cloned[:, y:y+patch_size, x:x+patch_size] = blended_patch
                
                modified_images.append(img_i_cloned)
                
                cx, cy = (x + patch_size / 2.0) / w, (y + patch_size / 2.0) / h
                pw, ph = patch_size / w, patch_size / h
                
                new_bboxes.append(torch.tensor([[cx, cy, pw, ph]], device=images.device))
                new_cls.append(torch.tensor([[target_class]], device=images.device))
                new_batch_idx.append(torch.tensor([i], device=images.device))
                is_poisoned = True
            else:
                modified_images.append(img_i)

        if is_poisoned:
            batch['img'] = torch.stack(modified_images, dim=0)
            batch['bboxes'] = torch.cat([bboxes] + new_bboxes, dim=0)
            batch['cls'] = torch.cat([cls] + new_cls, dim=0)
            batch['batch_idx'] = torch.cat([batch_idx] + new_batch_idx, dim=0)
            
        batch['is_poisoned'] = is_poisoned
        return batch

    def train_step(self, batch):
        # 오염된 데이터가 들어온 경우에만 2단계 교대 최적화 수행
        if self.is_attacker and self.attack_config.get("attack-flag", False) and batch.get('is_poisoned', False):
            
            # ==========================================
            # [Phase 1] 트리거 최적화 (모델 고정)
            # ==========================================
            self.model.requires_grad_(False)
            self.trigger_opt.zero_grad()
            
            # 첫 번째 포워드 패스로 트리거의 Loss 계산
            loss, _ = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.trigger_opt)
            self.trigger_opt.step()
            
            # 진화한 트리거 저장
            save_path = os.path.join(self.client_dir or ".", "trigger_patch.pt")
            torch.save(self.trigger_patch.detach().cpu(), save_path)

            # ==========================================
            # [Phase 2] 모델 최적화 (트리거 고정)
            # ==========================================
            self.model.requires_grad_(True)
            
            # [핵심 2] 1단계에서 소비된 연산 그래프를 끊어 PyTorch 에러 방지
            batch['img'] = batch['img'].detach()

        # 정상 데이터이거나 Phase 2일 때는 기존 YOLO의 학습 스텝 수행
        return super().train_step(batch)