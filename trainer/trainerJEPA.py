# %%writefile /content/DeepCAD_prashant/trainer/trainerJEPA.py
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
from model.collapse_monitor import CollapseMonitor


class TrainerJEPA:
    def __init__(self, cfg):
        self.cfg              = cfg
        self.global_step      = 0
        self.is_hierarchical  = (cfg.masking_strategy == 'hierarchical')
        self._build_models()
        self._build_optimizer()
        self.masker = get_masker(cfg.masking_strategy, cfg)
        self.monitor = CollapseMonitor(cfg.d_model, rank_threshold=0.70)
        self._max_rank_seen = 0.0
        self._rank_burn_in  = 20   # don't warn before epoch 20
        self._load_monitor_sample()

    # ──────────────────────────────────────────────────────────
    #  Build
    # ──────────────────────────────────────────────────────────

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

        n_enc  = sum(p.numel() for p in self.encoder.parameters())
        n_pred = sum(p.numel() for p in self.predictor.parameters())
        print(f"  Encoder params : {n_enc/1e6:.2f}M")
        print(f"  Predictor params: {n_pred/1e6:.2f}M")

    def _build_optimizer(self):
        params = (list(self.encoder.parameters()) +
                  list(self.predictor.parameters()))
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=0.05, betas=(0.9, 0.95)
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

    # ──────────────────────────────────────────────────────────
    #  EMA update with ramp
    # ──────────────────────────────────────────────────────────

    def _update_ema(self):
        # Ramp EMA decay from 0.99 → target over first 1000 steps
        t     = min(1.0, self.global_step / 1000)
        decay = 0.99 + (self.cfg.ema_decay - 0.99) * t
        self.ema.update(self.encoder, decay=decay)

    def _load_monitor_sample(self, n=512):
        """
        Load a fixed sample of validation sequences for epoch-level rank monitoring.
        Stored on CPU — moved to GPU only during the rank computation call.
        Uses batch_size=n to load in one shot.
        """
        from dataset.cad_dataset import get_dataloader

        class _Cfg:
            pass
        cfg             = _Cfg()
        cfg.data_root   = self.cfg.data_root
        cfg.batch_size  = n
        cfg.num_workers = 0      # no workers — this is a one-time load
        cfg.augment     = False
        cfg.max_n_loops  = self.cfg.max_n_loops
        cfg.max_n_curves = self.cfg.max_n_curves
        cfg.max_total_len = self.cfg.max_total_len
        cfg.max_n_ext    = self.cfg.max_n_ext

        loader = get_dataloader('validation', cfg, shuffle=True)
        batch  = next(iter(loader))
        self.monitor_cmds = batch['command'][:n]   # (N, 60) long, CPU
        self.monitor_args = batch['args'][:n]      # (N, 60, 16) long, CPU
        print(f"  Collapse monitor sample: {self.monitor_cmds.shape[0]} val sequences")

        
    # ──────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────

    def _forward(self, data):
        commands = data['command'].cuda()   # (N, S)
        args     = data['args'].cuda()      # (N, S, n_args)

        # ── Masking ──────────────────────────────────────────
        if self.is_hierarchical:
            target_mask, level_per_seq = self.masker(commands)
        else:
            target_mask    = self.masker(commands)
            level_per_seq  = None

        if target_mask.sum() == 0:
            return torch.tensor(0.0, device='cuda', requires_grad=True)

        # ── Context encoder ──────────────────────────────────
        ctx_emb = self.encoder(commands, args, target_mask=target_mask)  # (S,N,d)

        # ── EMA target encoder (no grad) ─────────────────────
        with torch.no_grad():
            full_emb = self.ema(commands, args)   # (S, N, d)

        # ── Key padding mask for predictor ───────────────────
        commands_sf      = commands.permute(1, 0)                          # (S, N)
        key_padding_mask = _get_key_padding_mask(commands_sf, seq_dim=0)   # (N, S)

        # ── Predict + loss ───────────────────────────────────
        if self.is_hierarchical:
            loss = self._hierarchical_loss(
                ctx_emb, full_emb, target_mask,
                key_padding_mask, level_per_seq
            )
        else:
            pred_all       = self.predictor(ctx_emb, key_padding_mask)    # (S,N,d)
            target_sf      = target_mask.permute(1, 0)                    # (S, N)
            pred_at_target = pred_all[target_sf]
            true_at_target = full_emb[target_sf]

            # if self.cfg.target_norm:
            #     pred_at_target = F.normalize(pred_at_target, dim=-1)
            #     true_at_target = F.normalize(true_at_target, dim=-1)

            # loss = F.smooth_l1_loss(pred_at_target, true_at_target.detach())

            # Cast to float32 before loss — critical when autocast (bfloat16) is active
            if self.cfg.target_norm:
                pred_at_target = F.normalize(pred_at_target.float(), dim=-1)
                true_at_target = F.normalize(true_at_target.float(), dim=-1)
            else:
                pred_at_target = pred_at_target.float()
                true_at_target = true_at_target.float()

            loss = F.smooth_l1_loss(pred_at_target, true_at_target.detach())

        return loss

    def _hierarchical_loss(self, ctx_emb, full_emb, target_mask,
                           key_padding_mask, level_per_seq):
        """
        Route each sequence to its corresponding predictor head,
        compute per-level loss, average across levels.
        """
        losses = []
        for level in ['token', 'block', 'group']:
            idx = [i for i, l in enumerate(level_per_seq) if l == level]
            if not idx:
                continue

            idx_t    = torch.tensor(idx, device='cuda')
            ctx_lv   = ctx_emb[:, idx_t, :]         # (S, n_lv, d)
            kpm_lv   = key_padding_mask[idx_t, :]   # (n_lv, S)
            full_lv  = full_emb[:, idx_t, :]        # (S, n_lv, d)
            mask_lv  = target_mask[idx_t, :]        # (n_lv, S)

            pred     = self.predictor(ctx_lv, level, kpm_lv)  # (S, n_lv, d)

            mask_sf  = mask_lv.permute(1, 0)         # (S, n_lv)
            pred_at  = pred[mask_sf]
            true_at  = full_lv[mask_sf]

            # if self.cfg.target_norm:
            #     pred_at = F.normalize(pred_at, dim=-1)
            #     true_at = F.normalize(true_at, dim=-1)

            # losses.append(F.smooth_l1_loss(pred_at, true_at.detach()))
            if self.cfg.target_norm:
                pred_at = F.normalize(pred_at.float(), dim=-1)
                true_at = F.normalize(true_at.float(), dim=-1)
            else:
                pred_at = pred_at.float()
                true_at = true_at.float()

            losses.append(F.smooth_l1_loss(pred_at, true_at.detach()))

        if not losses:
            return torch.tensor(0.0, device='cuda', requires_grad=True)
        return torch.stack(losses).mean()

    # ──────────────────────────────────────────────────────────
    #  Train / val loops
    # ──────────────────────────────────────────────────────────

    def _train_epoch(self, loader, epoch):
        self.encoder.train()
        self.predictor.train()
        losses = []
        pbar   = tqdm(loader, desc=f'Ep{epoch:04d}', leave=False)

        for data in pbar:
            self.optimizer.zero_grad()
            # loss = self._forward(data)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self._forward(data)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()),
                self.cfg.grad_clip
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self._update_ema()

            self.global_step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.5f}')

        return float(np.mean(losses))

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.encoder.eval()
        self.predictor.eval()
        losses = []
        for data in tqdm(loader, desc='Val', leave=False):
            losses.append(self._forward(data).item())
        return float(np.mean(losses))

    # ──────────────────────────────────────────────────────────
    #  Checkpoint
    # ──────────────────────────────────────────────────────────

    def save_ckpt(self, epoch, tag=None):
        name = tag or f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':        epoch,
            'global_step':  self.global_step,
            'encoder':      self.encoder.state_dict(),
            'ema_encoder':  self.ema.encoder.state_dict(),
            'predictor':    self.predictor.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
        }, path)
        return path

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cuda')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.ema.encoder.load_state_dict(ckpt['ema_encoder'])
        self.predictor.load_state_dict(ckpt['predictor'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    # ──────────────────────────────────────────────────────────
    #  Main train loop
    # ──────────────────────────────────────────────────────────

    def train(self):
        cfg          = self.cfg
        train_loader = get_dataloader('train',      cfg)
        val_loader   = get_dataloader('validation', cfg)

        total_steps  = cfg.nr_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        start_epoch = 0
        if cfg.cont:
            ckpt_path   = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'Resumed from epoch {start_epoch}')

        log_path = os.path.join(cfg.log_dir, 'losses.txt')
        best_val = float('inf')

        for epoch in range(start_epoch, cfg.nr_epochs):
            train_loss = self._train_epoch(train_loader, epoch)

            val_loss = 0.0
            if epoch % cfg.val_frequency == 0:
                val_loss = self._val_epoch(val_loader)

            lr_now = self.optimizer.param_groups[0]['lr']
            msg = (f'ep={epoch:04d}  train={train_loss:.5f}  '
                   f'val={val_loss:.5f}  lr={lr_now:.2e}  '
                   f'step={self.global_step}')
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

            # ── Rank monitoring (epoch level) ──────────────────
            rank = self.monitor.compute_rank(
                self.ema.encoder, self.monitor_cmds, self.monitor_args
            )
            self._max_rank_seen = max(self._max_rank_seen, rank)
            # Relative collapse: rank drops below 50% of best seen, after burn-in
            collapsing = (
                epoch >= self._rank_burn_in and
                self._max_rank_seen > 0 and
                rank < 0.50 * self._max_rank_seen
            )
            rank_msg = (f'  rank={rank:.3f}  '
                        f'max={self._max_rank_seen:.3f}  '
                        f'collapse={"⚠ YES" if collapsing else "no"}')
            print(rank_msg)
            with open(log_path, 'a') as f:
                f.write(rank_msg + '\n')
            if collapsing:
                print(f'  *** COLLAPSE WARNING ep={epoch} '
                      f'rank={rank:.3f} dropped below '
                      f'50% of max={self._max_rank_seen:.3f} ***')


            # Save every save_frequency epochs (needed for pretraining dynamics)
            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)

            # Always save latest (for resume)
            self.save_ckpt(epoch, tag='latest')

        print('Training complete.')