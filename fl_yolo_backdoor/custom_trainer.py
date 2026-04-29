from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import random
import os
import copy

FL_LOG_ROOT = os.environ.get(
    "FL_LOG_ROOT",
    "/home/flba/project/flwr_yolov8_lisa_template/fl_logs"
)
GLOBAL_TRIGGER_PATH = os.path.join(FL_LOG_ROOT, "global_trigger.pt")

class AnywhereDoorTrainer(DetectionTrainer):
    is_attacker = False
    attack_config = {}
    trigger_patch = None
    trigger_opt = None
    pid = None
    client_dir = None
    server_round = 0  

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_total_selected = 0
        self.debug_total_label_changed = 0
        self.debug_total_poisoned_batches = 0
        
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            self._init_trigger()

    def _init_trigger(self):
        patch_size = self.attack_config.get("trigger-size", 32)
        
        if not hasattr(self, "_debug_init_trigger_printed"):
            print(f"[DEBUG] init_trigger 호출됨: attacker={self.is_attacker}, round={self.server_round}")
            self._debug_init_trigger_printed = True

        if os.path.exists(GLOBAL_TRIGGER_PATH):
            try:
                init_tensor = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu", weights_only=True).clone()
            except TypeError:
                init_tensor = torch.load(GLOBAL_TRIGGER_PATH, map_location="cpu").clone()
                
            if getattr(self, "_debug_init_trigger_printed", False) is True and not hasattr(self, "_debug_trigger_loaded_msg"):
                print(f"[DEBUG] 글로벌 트리거 로드 완료 (Adaptive 모드): round={self.server_round}")
                self._debug_trigger_loaded_msg = True
        else:
            print("[WARN] 글로벌 트리거가 없어 임시 생성합니다.")
            generator = torch.Generator().manual_seed(42)
            init_tensor = torch.randn(3, patch_size, patch_size, generator=generator)

        self.trigger_patch = torch.nn.Parameter(init_tensor)
        self.trigger_opt = torch.optim.Adam([self.trigger_patch], lr=0.1)

    def _clone_batch(self, batch):
        cloned = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                cloned[k] = v.clone()
            else:
                cloned[k] = copy.deepcopy(v)
        return cloned

    def preprocess_batch(self, batch):
        if not hasattr(self, "_debug_preprocess_printed"):
            print("[DEBUG] preprocess_batch called (Adaptive Trigger 최적화 활성화)")
            self._debug_preprocess_printed = True
            
        batch = super().preprocess_batch(batch)
        
        if not (self.is_attacker and self.attack_config.get("attack-flag", False)):
            return batch

        if self.trigger_patch.device != batch['img'].device:
            self.trigger_patch.data = self.trigger_patch.data.to(batch['img'].device)

        trigger_batch = self._clone_batch(batch)
        
        poisoned_batch_for_trigger, poison_mask = self._inject_anywhere_door(trigger_batch, detach_trigger=False)

        if poisoned_batch_for_trigger.get("is_poisoned", False):
            self._optimize_trigger_on_batch(poisoned_batch_for_trigger)
            final_poisoned_batch, _ = self._inject_anywhere_door(batch, detach_trigger=True, force_mask=poison_mask)
            return final_poisoned_batch
            
        return batch

    def _set_bn_eval(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    def _to_scalar_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            scalar = None
            for item in loss:
                if torch.is_tensor(item):
                    val = item.sum()
                else:
                    print(f"[WARN] Non-tensor loss item received: {type(item)}")
                    val = torch.tensor(float(item), device=self.trigger_patch.device)
                scalar = val if scalar is None else scalar + val
            return scalar

        if torch.is_tensor(loss):
            return loss.sum()

        print(f"[WARN] Non-tensor loss received: {type(loss)}")
        return torch.tensor(float(loss), device=self.trigger_patch.device, requires_grad=True)

    def _optimize_trigger_on_batch(self, poisoned_batch):
        if not hasattr(self, "trigger_opt") or self.trigger_opt is None:
            return

        was_training = self.model.training
        grad_states = [p.requires_grad for p in self.model.parameters()]

        self.model.train()
        self._set_bn_eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        inner_steps = int(self.attack_config.get("trigger-inner-steps", 1))

        with torch.enable_grad():
            for _ in range(inner_steps):
                self.trigger_opt.zero_grad(set_to_none=True)
                
                outputs = self.model(poisoned_batch)
                
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                    loss_items = outputs[1] if len(outputs) > 1 else None
                else:
                    loss = outputs
                    loss_items = None

                if not hasattr(self, "_debug_loss_shape_printed"):
                    print(
                        f"[DEBUG] trigger loss type={type(loss)}, "
                        f"shape={getattr(loss, 'shape', None)}, "
                        f"loss_items_shape={getattr(loss_items, 'shape', None)}"
                    )
                    self._debug_loss_shape_printed = True

                loss = self._to_scalar_loss(loss)
                loss.backward()

                if not hasattr(self, "_debug_trigger_grad_printed"):
                    grad = self.trigger_patch.grad
                    print(
                        "[DEBUG] trigger grad:",
                        None if grad is None else {
                            "mean": grad.abs().mean().item(),
                            "max": grad.abs().max().item()
                        }
                    )
                    self._debug_trigger_grad_printed = True
                    
                self.trigger_opt.step()

        for p, req in zip(self.model.parameters(), grad_states):
            p.requires_grad_(req)

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        if self.client_dir:
            save_path = os.path.join(self.client_dir, f"trigger_patch_round_{self.server_round}.pt")
            torch.save(self.trigger_patch.detach().cpu(), save_path)

    def _inject_anywhere_door(self, batch, detach_trigger: bool = True, force_mask: list = None):
        images = batch["img"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"]

        b, c, h, w = images.shape
        poison_rate = float(self.attack_config.get("poison-rate", 1.0))
        target_class = int(self.attack_config.get("target-class", 0))
        source_class = self.attack_config.get("source-class", None)
        source_class = None if source_class is None else int(source_class)
        epsilon = float(self.attack_config.get("epsilon", 0.05))

        modified_images = []
        is_poisoned = False
        new_poison_mask = []

        patch_size = self.trigger_patch.shape[2]
        tile_h = (h + patch_size - 1) // patch_size
        tile_w = (w + patch_size - 1) // patch_size
        mosaicked_trigger = self.trigger_patch.repeat(1, tile_h, tile_w)[:, :h, :w]
        
        trigger_noise = epsilon * (2.0 * torch.sigmoid(mosaicked_trigger) - 1.0)
        if detach_trigger:
            trigger_noise = trigger_noise.detach() 

        cls_view = cls.view(-1)
        cls_long = cls_view.long()
        batch_idx_view = batch_idx.view(-1).long()

        selected_count = 0
        label_changed_count = 0

        for i in range(b):
            img_i = images[i]
            
            if force_mask is not None:
                do_poison = force_mask[i]
            else:
                do_poison = (random.random() < poison_rate)

            if not do_poison:
                new_poison_mask.append(False)
                modified_images.append(img_i)
                continue

            img_obj_mask = batch_idx_view == i
            if source_class is not None:
                obj_mask = img_obj_mask & (cls_long == source_class)
            else:
                obj_mask = img_obj_mask & (cls_long != target_class)

            if obj_mask.sum().item() == 0:
                new_poison_mask.append(False)
                modified_images.append(img_i)
                continue

            new_poison_mask.append(True)
            selected_count += 1
            cls_view[obj_mask] = float(target_class)
            label_changed_count += int(obj_mask.sum().item())
            
            img_i_cloned = torch.clamp(img_i + trigger_noise, 0.0, 1.0)
            modified_images.append(img_i_cloned)
            is_poisoned = True

        if is_poisoned:
            batch["img"] = torch.stack(modified_images, dim=0)
            batch["cls"] = cls_view.view_as(cls)
        batch["is_poisoned"] = is_poisoned
        
        if detach_trigger:
            self.debug_total_selected += selected_count
            self.debug_total_label_changed += label_changed_count
            self.debug_total_poisoned_batches += int(is_poisoned)

            if not hasattr(self, "_debug_poison_printed") and selected_count > 0:
                print(f"[DEBUG] first poison batch: selected(real)={selected_count}, label_changed_objects={label_changed_count}, is_poisoned={is_poisoned}")
                self._debug_poison_printed = True
            
        return batch, new_poison_mask