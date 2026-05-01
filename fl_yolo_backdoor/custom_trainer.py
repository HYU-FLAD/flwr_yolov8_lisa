from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import torch.nn as nn
import random
import os
import copy

FL_LOG_ROOT = os.environ.get("FL_LOG_ROOT", "/home/flba/project/flwr_yolov8_lisa_template/fl_logs")
GLOBAL_GEN_PATH = os.path.join(FL_LOG_ROOT, "global_generator.pt")

class AnywhereDoorGenerator(nn.Module):
    def __init__(self, num_classes=3, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        self.G_r = nn.Sequential(
            nn.Linear(num_classes, 128), nn.ReLU(),
            nn.Linear(128, 3 * patch_size * patch_size)
        )
        self.G_g = nn.Sequential(
            nn.Linear(num_classes, 128), nn.ReLU(),
            nn.Linear(128, 3 * patch_size * patch_size)
        )

    def forward(self, e_r, e_g):
        batch_size = e_r.size(0)
        
        out_r = self.G_r(e_r).view(batch_size, 3, self.patch_size, self.patch_size)
        out_g = self.G_g(e_g).view(batch_size, 3, self.patch_size, self.patch_size)
        
        r_active = (e_r.sum(dim=1) > 0).float().view(batch_size, 1, 1, 1)
        g_active = (e_g.sum(dim=1) > 0).float().view(batch_size, 1, 1, 1)

        out_r = out_r * r_active
        out_g = out_g * g_active
        
        trigger = out_r + out_g
        return trigger

class AnywhereDoorTrainer(DetectionTrainer):
    is_attacker = False
    attack_config = {}
    generator = None
    gen_opt = None
    pid = None
    client_dir = None
    server_round = 0  
    rng = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_total_selected = 0
        self.debug_total_label_changed = 0
        self.debug_total_poisoned_batches = 0
        
        seed_val = int(self.attack_config.get("seed", 0)) + int(self.server_round) * 100000
        self.rng = random.Random(seed_val)
        
        if self.is_attacker and self.attack_config.get("attack-flag", False):
            self._init_generator()

    def _init_generator(self):
        patch_size = int(self.attack_config.get("trigger-size", 32))
        nc = int(self.attack_config.get("num-classes", 3))
        
        self.generator = AnywhereDoorGenerator(num_classes=nc, patch_size=patch_size)
        
        if os.path.exists(GLOBAL_GEN_PATH):
            try: state_dict = torch.load(GLOBAL_GEN_PATH, map_location="cpu", weights_only=True)
            except TypeError: state_dict = torch.load(GLOBAL_GEN_PATH, map_location="cpu")
            self.generator.load_state_dict(state_dict)
            if not hasattr(self, "_debug_gen_loaded"):
                print(f"[DEBUG] Global Generator G_phi Loaded (Round {self.server_round}, nc={nc})")
                self._debug_gen_loaded = True
        else:
            print("[WARN] Global Generator가 없어 초기화합니다.")
            
        gen_lr = float(self.attack_config.get("generator-lr", 0.01))
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        
        if self.device: self.generator.to(self.device)

    def _clone_batch(self, batch):
        cloned = {}
        for k, v in batch.items():
            cloned[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
        return cloned

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not hasattr(self, "_debug_preprocess_printed"):
            print("[DEBUG] preprocess_batch called (True Adaptive Trigger 최적화)")
            self._debug_preprocess_printed = True
            
        if not (self.is_attacker and self.attack_config.get("attack-flag", False)): 
            return batch

        probe_batch = self._clone_batch(batch)
        poisoned_probe, e_info, mask_info = self._inject_anywhere_door(
            probe_batch, 
            detach_trigger=False
        )
        
        is_poisoned = bool(poisoned_probe.get("is_poisoned", False))
        del poisoned_probe, probe_batch

        if is_poisoned:
            self._optimize_generator_on_batch(
                base_batch=batch,
                e_info=e_info,
                mask_info=mask_info
            )
            
            final_batch, _, _ = self._inject_anywhere_door(
                batch, 
                detach_trigger=True, 
                force_mask=mask_info, 
                force_e=e_info
            )
            return final_batch
            
        return batch

    def _set_bn_eval(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.eval()

    def _to_scalar_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            scalar = None
            for item in loss:
                val = item.sum() if torch.is_tensor(item) else torch.tensor(float(item), device=self.device)
                scalar = val if scalar is None else scalar + val
            return scalar
        if torch.is_tensor(loss): return loss.sum()
        print(f"[WARN] Non-tensor loss received: {type(loss)}")
        return torch.tensor(float(loss), device=self.device, requires_grad=True)

    def _optimize_generator_on_batch(self, base_batch, e_info, mask_info):
        if not hasattr(self, "gen_opt") or self.gen_opt is None:
            return

        was_training = self.model.training
        grad_states = [p.requires_grad for p in self.model.parameters()]

        try:
            self.model.train()
            self._set_bn_eval()

            for p in self.model.parameters():
                p.requires_grad_(False)

            self.generator.train()
            inner_steps = int(self.attack_config.get("trigger-inner-steps", 1))

            with torch.enable_grad():
                for _ in range(inner_steps):
                    step_batch = self._clone_batch(base_batch)
                    poisoned_step, _, _ = self._inject_anywhere_door(
                        step_batch,
                        detach_trigger=False,
                        force_mask=mask_info,
                        force_e=e_info
                    )

                    self.gen_opt.zero_grad(set_to_none=True)
                    outputs = self.model(poisoned_step)
                    loss = self._to_scalar_loss(outputs[0] if isinstance(outputs, tuple) else outputs)
                    loss.backward()

                    grad_norm = 0.0
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().abs().sum().item()

                    if not hasattr(self, "_debug_trigger_grad_printed"):
                        print(f"[DEBUG] G_phi grad_norm={grad_norm:.6f}")
                        if grad_norm <= 0:
                            print("[FATAL] Generator G_phi gradient norm is zero.")
                        self._debug_trigger_grad_printed = True

                    self.gen_opt.step()

                    del outputs, loss, poisoned_step, step_batch

        finally:
            for p, req in zip(self.model.parameters(), grad_states):
                p.requires_grad_(req)

            if was_training:
                self.model.train()
            else:
                self.model.eval()

            if hasattr(self, "generator") and self.generator is not None:
                self.generator.eval()

        if self.client_dir:
            save_path = os.path.join(self.client_dir, f"generator_round_{self.server_round}.pt")
            torch.save(self.generator.state_dict(), save_path)

    def _inject_anywhere_door(self, batch, detach_trigger: bool, force_mask=None, force_e=None):
        images, cls, batch_idx = batch["img"], batch["cls"], batch["batch_idx"]
        b, c, h, w = images.shape
        
        poison_rate = float(self.attack_config.get("poison-rate", 0.7))
        epsilon = float(self.attack_config.get("epsilon", 0.10))
        nc = self.generator.num_classes

        modified_images, new_mask, e_info = [], [], []
        is_poisoned, selected, lbl_chg = False, 0, 0

        cls_view = cls.view(-1)
        cls_long = cls_view.long()
        batch_idx_view = batch_idx.view(-1).long()
        keep_indices = []

        for i in range(b):
            img_i = images[i]
            img_obj_mask = (batch_idx_view == i)
            
            do_poison = force_mask[i] if force_mask is not None else (self.rng.random() < poison_rate)

            if not do_poison or img_obj_mask.sum().item() == 0:
                new_mask.append(False)
                e_info.append(None)
                modified_images.append(img_i)
                keep_indices.append(img_obj_mask.nonzero(as_tuple=True)[0])
                continue

            if force_e is not None and force_e[i] is not None:
                src_c, tgt_c, attack_type = force_e[i]
            else:
                available_classes = [int(x) for x in cls_long[img_obj_mask].unique().tolist() if 0 <= int(x) < nc]
                
                fixed_src = int(self.attack_config.get("fixed-source-class", 0))
                fixed_tgt = int(self.attack_config.get("fixed-target-class", 1))

                # [추가됨] 방어 코드: source와 target이 같으면 에러 발생
                if fixed_src == fixed_tgt:
                    raise ValueError(f"fixed-source-class and fixed-target-class must differ: {fixed_src}")

                if fixed_src not in available_classes:
                    new_mask.append(False)
                    e_info.append(None)
                    modified_images.append(img_i)
                    keep_indices.append(img_obj_mask.nonzero(as_tuple=True)[0])
                    continue
                
                src_c = fixed_src
                tgt_c = fixed_tgt
                attack_type = "targeted_miscls"

            source_obj_mask = img_obj_mask & (cls_long == src_c)

            if source_obj_mask.sum().item() == 0:
                new_mask.append(False)
                e_info.append(None)
                modified_images.append(img_i)
                keep_indices.append(img_obj_mask.nonzero(as_tuple=True)[0])
                continue

            new_mask.append(True)
            e_info.append((src_c, tgt_c, attack_type))
            selected += 1
            is_poisoned = True

            e_r_vec = torch.zeros(nc, device=self.device)
            e_g_vec = torch.zeros(nc, device=self.device)
            
            e_r_vec[src_c] = 1.0 
            if attack_type == "targeted_miscls":
                e_g_vec[tgt_c] = 1.0 

            trigger_patch = self.generator(e_r_vec.unsqueeze(0), e_g_vec.unsqueeze(0)).squeeze(0)
            
            patch_size = trigger_patch.shape[1]
            tile_h, tile_w = (h + patch_size - 1) // patch_size, (w + patch_size - 1) // patch_size
            mosaicked = trigger_patch.repeat(1, tile_h, tile_w)[:, :h, :w]
            trigger_noise = epsilon * (2.0 * torch.sigmoid(mosaicked) - 1.0)
            
            if detach_trigger: trigger_noise = trigger_noise.detach()
            modified_images.append(torch.clamp(img_i + trigger_noise, 0.0, 1.0))

            if attack_type == "targeted_miscls":
                cls_view[source_obj_mask] = float(tgt_c)
                lbl_chg += int(source_obj_mask.sum().item())
                keep_indices.append(img_obj_mask.nonzero(as_tuple=True)[0])
            elif attack_type == "targeted_removal":
                lbl_chg += int(source_obj_mask.sum().item())
                keep_indices.append((img_obj_mask & ~source_obj_mask).nonzero(as_tuple=True)[0])

        batch["img"] = torch.stack(modified_images, dim=0)
        
        if len(keep_indices) > 0:
            final_keep = torch.cat(keep_indices, dim=0)
            batch["cls"] = cls_view[final_keep].view(-1, 1)
            batch["batch_idx"] = batch_idx_view[final_keep]
            if "bboxes" in batch and batch["bboxes"] is not None: 
                batch["bboxes"] = batch["bboxes"][final_keep]
        else:
            batch["cls"] = torch.empty((0, 1), device=cls.device)
            batch["batch_idx"] = torch.empty((0,), dtype=torch.long, device=batch_idx.device)
            if "bboxes" in batch and batch["bboxes"] is not None: 
                batch["bboxes"] = torch.empty((0, 4), device=batch["bboxes"].device)

        batch["is_poisoned"] = is_poisoned
        if detach_trigger:
            self.debug_total_selected += selected
            self.debug_total_label_changed += lbl_chg
            self.debug_total_poisoned_batches += int(is_poisoned)
            
        return batch, e_info, new_mask