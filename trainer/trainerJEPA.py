import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.jepa_encoder import JEPAEncoder
from model.ema_target import EMATargetEncoder
from model.predictor import JEPAPredictor, HierarchicalPredictor
from model.masker import get_masker
from model.model_utils import _get_key_padding_mask
from dataset.cad_dataset import get_dataloader
from utils import ensure_dirs


class TrainerJEPA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0
        self.is_hierarchical = (cfg.masking_strategy == 'hierarchical')

        self._build_models()
        self._build_optimizer()
        self.masker = get_masker(cfg.masking_strategy, cfg)

    # ──────────────────────────────────────────
    #  Build
    # ──────────────────────────────────────────

    def _build_models(self):
        cfg = self.cfg

        self.encoder = JEPAEncoder(cfg).cuda()
        self.ema     = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)
        self.ema.encoder = self.ema.encoder.cuda()

        pred_kwargs = dict(
            d_model        = cfg.d_model,
            pred_dim       = cfg.pred_dim,
            n_layers       = cfg.pred_depth,
            n_heads        = cfg.pred_heads,
            dim_feedforward= cfg.pred_ffn_dim,
            dropout        = cfg.dropout,
        )
        if self.is_hierarchical:
            self.predictor = HierarchicalPredictor(**pred_kwargs).cuda()
        else:
            self.predictor = JEPAPredictor(**pred_kwargs).cuda()

    def _build_optimizer(self):
        cfg = self.cfg
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.05,
                                           betas=(0.9, 0.95))
        # Scheduler set after total_steps is known
        self.scheduler = None

    def _build_scheduler(self, total_steps):
        warmup = self.cfg.warmup_step
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ──────────────────────────────────────────
    #  Forward step
    # ──────────────────────────────────────────

    def _forward(self, data):
        commands = data['command'].cuda()   # (N, S)
        args     = data['args'].cuda()      # (N, S, n_args)

        if self.is_hierarchical:
            target_mask, level_per_seq = self.masker(commands)
            # For hierarchical: route each sequence to correct predictor head
            # But since sequences in a batch may have different levels,
            # we use the MAJORITY level for the predictor head this step.
            # (All sequences share one forward pass through the encoder.)
            level = max(set(level_per_seq), key=level_per_seq.count)
        else:
            target_mask = self.masker(commands)
            level = None

        if target_mask.sum() == 0:
            return torch.tensor(0.0, device='cuda', requires_grad=True)

        # 2. Context encoder (sees masked sequence)
        ctx_emb = self.encoder(commands, args, target_mask=target_mask)  # (S, N, d_model)

        # 3. EMA target encoder (sees full sequence, no grad)
        with torch.no_grad():
            full_emb = self.ema(commands, args)                          # (S, N, d_model)

        # 4. Key padding mask for predictor (based on original commands)
        commands_sf      = commands.permute(1, 0)                        # (S, N)
        key_padding_mask = _get_key_padding_mask(commands_sf, seq_dim=0) # (N, S)

        # 5. Predictor
        if self.is_hierarchical:
            pred_all = self.predictor(ctx_emb, level, key_padding_mask)
        else:
            pred_all = self.predictor(ctx_emb, key_padding_mask)         # (S, N, d_model)

        # 6. Extract target positions and compute loss
        target_mask_sf = target_mask.permute(1, 0)   # (S, N)
        pred_at_target = pred_all[target_mask_sf]    # (n_masked, d_model)
        true_at_target = full_emb[target_mask_sf]    # (n_masked, d_model)

        if self.cfg.target_norm:
            true_at_target = F.normalize(true_at_target, dim=-1)
            pred_at_target = F.normalize(pred_at_target, dim=-1)

        loss = F.smooth_l1_loss(pred_at_target, true_at_target.detach())
        return loss

    # ──────────────────────────────────────────
    #  EMA update with cosine warmup schedule
    # ──────────────────────────────────────────

    def _update_ema(self):
        # EMA decay ramps from 0 → cfg.ema_decay in first 10k steps
        decay = min(self.cfg.ema_decay,
                    (1 + self.global_step) / (10 + self.global_step))
        self.ema.update(self.encoder, decay=decay)

    # ──────────────────────────────────────────
    #  Train / val loops
    # ──────────────────────────────────────────

    def _train_epoch(self, loader, epoch):
        self.encoder.train()
        self.predictor.train()
        losses = []

        pbar = tqdm(loader, desc=f'Train ep{epoch}', leave=False)
        for data in pbar:
            self.optimizer.zero_grad()
            loss = self._forward(data)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()),
                self.cfg.grad_clip
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self._update_ema()

            self.global_step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        return np.mean(losses)

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.encoder.eval()
        self.predictor.eval()
        losses = []
        for data in tqdm(loader, desc='Val', leave=False):
            losses.append(self._forward(data).item())
        return np.mean(losses)

    # ──────────────────────────────────────────
    #  Checkpoint
    # ──────────────────────────────────────────

    def save_ckpt(self, epoch, tag=None):
        name = tag if tag else f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':       epoch,
            'global_step': self.global_step,
            'encoder':     self.encoder.state_dict(),
            'ema_encoder': self.ema.encoder.state_dict(),
            'predictor':   self.predictor.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
        }, path)
        print(f'  Saved: {path}')

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cuda')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.ema.encoder.load_state_dict(ckpt['ema_encoder'])
        self.predictor.load_state_dict(ckpt['predictor'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    # ──────────────────────────────────────────
    #  Main train loop
    # ──────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        train_loader = get_dataloader('train',      cfg)
        val_loader   = get_dataloader('validation', cfg)

        total_steps = cfg.nr_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        start_epoch = 0
        if cfg.cont:
            ckpt_path = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'Resumed from epoch {start_epoch}')

        log_path = os.path.join(cfg.log_dir, 'losses.txt')

        for epoch in range(start_epoch, cfg.nr_epochs):
            train_loss = self._train_epoch(train_loader, epoch)

            val_loss = 0.0
            if epoch % cfg.val_frequency == 0:
                val_loss = self._val_epoch(val_loader)

            msg = f'[ep {epoch:04d}] train={train_loss:.5f}  val={val_loss:.5f}'
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

            # Save every save_frequency epochs (for pretraining dynamics curve)
            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)

            self.save_ckpt(epoch, tag='latest')

        print('Training done.')