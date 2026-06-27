# %%writefile /content/DeepCAD_prashant/trainer/trainerMAE.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.jepa_encoder import JEPAEncoder
from model.cad_mae_head import CADMAEHead, mae_loss
from model.masker_new import TokenLevelMasker
from dataset.cad_dataset_new import get_dataloader


class TrainerMAE:
    """
    MAE-CAD: identical encoder architecture to JEPA.
    Token-level masking → reconstruct command + args at masked positions.
    Head discarded after pre-training. Only encoder used for downstream eval.
    """
    def __init__(self, cfg):
        self.cfg         = cfg
        self.global_step = 0
        self._build_models()
        self._build_optimizer()
        self.masker = TokenLevelMasker(
            mask_ratio=getattr(cfg, 'mask_ratio_token', 0.50)
        )

    def _build_models(self):
        cfg           = self.cfg
        self.encoder  = JEPAEncoder(cfg).cuda()
        self.mae_head = CADMAEHead(cfg.d_model).cuda()
        n_enc  = sum(p.numel() for p in self.encoder.parameters())
        n_head = sum(p.numel() for p in self.mae_head.parameters())
        print(f"[MAE] Encoder : {n_enc/1e6:.2f}M")
        print(f"[MAE] MAE head: {n_head/1e6:.2f}M")

    def _build_optimizer(self):
        decay, no_decay, seen = [], [], set()
        for name, param in (list(self.encoder.named_parameters()) +
                             list(self.mae_head.named_parameters())):
            if id(param) in seen:
                continue
            seen.add(id(param))
            no_decay_names = (
                'bias', 'norm', 'embedding', 'embed', 'mask_embedding', 'cls_token'
            )
            if param.ndim <= 1 or any(k in name for k in no_decay_names):
                no_decay.append(param)
            else:
                decay.append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {'params': decay,    'weight_decay': 0.05},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.cfg.lr, betas=(0.9, 0.95)
        )
        self.scheduler = None

    def _build_scheduler(self, total_steps):
        warmup = self.cfg.warmup_step
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _forward(self, data):
        commands    = data['command'].cuda()
        args        = data['args'].cuda()
        target_mask = self.masker(commands)   # (N, S) bool
        if target_mask.sum() == 0:
            return torch.tensor(0.0, device='cuda', requires_grad=True)
        memory, _       = self.encoder(commands, args, target_mask=target_mask)
        cmd_logits, args_logits = self.mae_head(memory, target_mask)
        return mae_loss(cmd_logits, args_logits, commands, args, target_mask)

    def _train_epoch(self, loader):
        self.encoder.train()
        self.mae_head.train()
        losses = []
        pbar   = tqdm(loader, leave=False)
        for data in pbar:
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self._forward(data)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.mae_head.parameters()), self.cfg.grad_clip
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.global_step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.5f}')
        return float(np.mean(losses))

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.encoder.eval()
        self.mae_head.eval()
        losses = []
        for data in tqdm(loader, leave=False):
            losses.append(self._forward(data).item())
        return float(np.mean(losses))

    def _backup_to_drive(self, path, tag):
        drive_dir = getattr(self.cfg, 'drive_backup_dir', None)
        if drive_dir and os.path.exists(drive_dir):
            import shutil
            dest = os.path.join(drive_dir, os.path.basename(path))
            shutil.copy(path, dest)
            # print(f'  [Drive] Backed up → {dest}')

    def save_ckpt(self, epoch, tag=None):
        name = tag or f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':       epoch,
            'encoder':     self.encoder.state_dict(),
            'mae_head':    self.mae_head.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
        self._backup_to_drive(path, name)
        return path

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.mae_head.load_state_dict(ckpt['mae_head'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    def train(self):
        cfg          = self.cfg
        train_loader = get_dataloader('train', cfg)
        val_loader   = get_dataloader('validation', cfg)
        total_steps  = cfg.nr_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        start_epoch = 0
        if cfg.cont:
            ckpt_path   = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'[MAE] Resumed from epoch {start_epoch}')

        log_path = os.path.join(cfg.log_dir, 'losses.txt')

        for epoch in range(start_epoch, cfg.nr_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss   = None
            if epoch % cfg.val_frequency == 0:
                val_loss = self._val_epoch(val_loader)
            lr_now  = self.optimizer.param_groups[0]['lr']
            val_str = f'{val_loss:.5f}' if val_loss is not None else '-------'
            msg     = (f'ep={epoch:04d}  train={train_loss:.5f}  '
                       f'val={val_str}  lr={lr_now:.2e}  step={self.global_step}')
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)
            self.save_ckpt(epoch, tag='latest')

        print('[MAE] Training complete.')