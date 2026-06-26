import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.jepa_encoder import JEPAEncoder
from model.ema_target import EMATargetEncoder
from model.predictor import CADJEPAPredictor
from model.masker import get_masker
from model.model_utils import _get_key_padding_mask, _get_padding_mask
from model.collapse_monitor import CollapseMonitor
from dataset.cad_dataset import get_dataloader
from cadlib.macro import EOS_IDX


class TrainerJEPA:
    def __init__(self, cfg):
        self.cfg         = cfg
        self.global_step = 0
        self._build_models()
        self._build_optimizer()
        self.masker  = get_masker(cfg.masking_strategy, cfg)
        self.monitor = CollapseMonitor(cfg.d_model, rank_threshold=0.70)
        self._max_rank_seen = 0.0
        self._rank_burn_in  = 20
        self._load_monitor_sample()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_models(self):
        cfg = self.cfg

        self.encoder   = JEPAEncoder(cfg).cuda()
        self.ema       = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)
        self.ema.encoder = self.ema.encoder.cuda()

        self.predictor = CADJEPAPredictor(
            d_model         = cfg.d_model,
            pred_dim        = cfg.pred_dim,
            n_layers        = cfg.pred_depth,
            n_heads         = cfg.pred_heads,
            dim_feedforward = cfg.pred_ffn_dim,
            dropout         = 0.0,
            max_seq_len     = cfg.max_seq_len,
        ).cuda()

        n_enc  = sum(p.numel() for p in self.encoder.parameters())
        n_pred = sum(p.numel() for p in self.predictor.parameters())
        print(f"[Trainer] Encoder params  : {n_enc  / 1e6:.2f}M")
        print(f"[Trainer] Predictor params: {n_pred / 1e6:.2f}M")

    def _build_optimizer(self):
        """
        Standard transformer weight decay split:
          decay    — weight matrices
          no_decay — biases, norms, embeddings, 1D params
        """
        decay, no_decay, seen = [], [], set()

        for name, param in (list(self.encoder.named_parameters()) +
                             list(self.predictor.named_parameters())):
            if id(param) in seen:
                continue
            seen.add(id(param))

            no_decay_names = (
                'bias', 'norm', 'embedding', 'embed',
                'position_queries', 'mask_embedding', 'cls_token'
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

        n_d  = sum(p.numel() for p in decay)
        n_nd = sum(p.numel() for p in no_decay)
        print(f"[Trainer] Params with decay   : {n_d  / 1e6:.2f}M")
        print(f"[Trainer] Params without decay: {n_nd / 1e6:.2f}M")

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

    # ── EMA momentum schedule ─────────────────────────────────────────────────
    #
    # Four-phase schedule aligned to dataset characteristics:
    #
    #   Phase 1 (step 0 → ema_warmup_steps):
    #     Linear ramp 0.990 → 0.994
    #     Target encoder responds quickly — important when single-block token
    #     masking dominates (52% of data) and gradient signal is weak.
    #
    #   Phase 2 (warmup → 50% of training):
    #     Hold at 0.994 — stable target during block masking learning.
    #
    #   Phase 3 (50% → 75% of training):
    #     Ramp 0.994 → 0.997 — representations stabilizing.
    #
    #   Phase 4 (75% → end):
    #     Ramp 0.997 → ema_decay (default 0.999) — late consolidation.
    #
    # Never use 0.999 from step 0 on this dataset:
    # at batch=256, 375 steps/epoch, effective memory = 1/(1-0.999) = 1000 steps
    # = 2.7 epochs. With weak early gradient signal the target becomes stale.
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ema_momentum(self, total_steps):
        s         = self.global_step
        warmup    = self.cfg.ema_warmup_steps
        final_mom = self.cfg.ema_decay   # default 0.999

        if s < warmup:
            # Phase 1: linear ramp 0.990 → 0.994
            t = s / max(1, warmup)
            return 0.990 + (0.994 - 0.990) * t

        phase2_end = total_steps // 2
        phase3_end = total_steps * 3 // 4

        if s < phase2_end:
            # Phase 2: hold at 0.994
            return 0.994

        if s < phase3_end:
            # Phase 3: ramp 0.994 → 0.997
            t = (s - phase2_end) / max(1, phase3_end - phase2_end)
            return 0.994 + (0.997 - 0.994) * t

        # Phase 4: ramp 0.997 → final_mom
        t = (s - phase3_end) / max(1, total_steps - phase3_end)
        return 0.997 + (final_mom - 0.997) * t

    def _update_ema(self, total_steps):
        momentum = self._get_ema_momentum(total_steps)
        self.ema.update(self.encoder, decay=momentum)

    # ── Monitor sample ────────────────────────────────────────────────────────

    def _load_monitor_sample(self, n=512):
        """Fixed validation sample for collapse monitoring each epoch."""
        class _Cfg:
            pass
        c                  = _Cfg()
        c.data_root        = self.cfg.data_root
        c.batch_size       = n
        c.num_workers      = 0
        c.augment          = False
        c.jitter_aug       = False
        c.translate_aug    = False
        c.max_n_loops      = self.cfg.max_n_loops
        c.max_n_curves     = self.cfg.max_n_curves
        c.max_total_len    = self.cfg.max_total_len
        c.max_n_ext        = self.cfg.max_n_ext
        c.dedup_ids_path   = None   # val split — no dedup filter

        loader = get_dataloader('validation', c, shuffle=True)
        batch  = next(iter(loader))
        self.monitor_cmds = batch['command'][:n]
        self.monitor_args = batch['args'][:n]
        print(f"[Trainer] Collapse monitor: {self.monitor_cmds.shape[0]} val sequences")

    # ── Masked index extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_masked_indices(target_mask):
        """
        target_mask: (N, S) bool
        Returns:
            padded: (N, max_n) long — masked position indices, padded with last real
            n_real: (N,)       long — actual masked count per sequence
        """
        n_real = target_mask.sum(dim=1)
        max_n  = int(n_real.max().item())

        if max_n == 0:
            return None, n_real

        N      = target_mask.shape[0]
        padded = torch.zeros(N, max_n, dtype=torch.long, device=target_mask.device)

        for i in range(N):
            indices = target_mask[i].nonzero(as_tuple=False).squeeze(-1)
            n = indices.shape[0]
            if n > 0:
                padded[i, :n] = indices
                if n < max_n:
                    padded[i, n:] = indices[-1]   # pad with last real index

        return padded, n_real

    # ── Variance regularisation ───────────────────────────────────────────────

    def _variance_loss(self, ctx_emb_sf, commands, target_mask):
        """
        Variance regularisation on non-masked, non-EOS context token embeddings.
        Forces each embedding dimension to have std >= 1 across the token batch.
        Prevents slow variance collapse that EMA alone can miss on homogeneous data.

        Weight: vicreg_lambda_v = 0.05 (small — must not dominate JEPA loss).
        Covariance term off by default (vicreg_lambda_c = 0.0).
        Enable covariance only if collapse is detected.

        ctx_emb_sf:  (S, N, d_model) seq-first
        commands:    (N, S)          batch-first long
        target_mask: (N, S)          bool
        """
        is_eos    = (commands.permute(1, 0) == EOS_IDX)   # (S, N)
        is_masked = target_mask.permute(1, 0)              # (S, N)
        valid     = (~is_eos & ~is_masked)                 # true context tokens

        z = ctx_emb_sf.float()[valid]   # (n_valid, d)
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=ctx_emb_sf.device)

        n, d = z.shape
        z    = z - z.mean(dim=0, keepdim=True)

        # Variance term: push each dim to have std >= 1
        std    = torch.sqrt(z.var(dim=0) + 1e-4)
        v_loss = F.relu(1.0 - std).mean()

        total = self.cfg.vicreg_lambda_v * v_loss

        # Covariance term (off by default)
        if self.cfg.vicreg_lambda_c > 0.0:
            cov      = (z.T @ z) / max(n - 1, 1)
            off_diag = cov.pow(2)
            off_diag.fill_diagonal_(0.0)
            c_loss   = off_diag.sum() / d
            total    = total + self.cfg.vicreg_lambda_c * c_loss

        return total

    # ── Core prediction loss ──────────────────────────────────────────────────

    def _prediction_loss(self, ctx_for_pred, full_emb, target_mask, key_padding_mask):
        """
        Cross-attention predictor loss.
        Predictor attends over zeroed context to predict EMA target embeddings
        at masked positions.

        ctx_for_pred:     (S, N, d_model) — encoder output, masked positions zeroed
        full_emb:         (S, N, d_model) — EMA target encoder output (no grad)
        target_mask:      (N, S) bool
        key_padding_mask: (N, S) bool — True at EOS/padding
        """
        padded, n_real = self._extract_masked_indices(target_mask)
        if padded is None:
            return torch.tensor(0.0, device='cuda', requires_grad=True)

        # Cross-attention: position queries attend to context
        pred_sf = self.predictor(
            ctx_for_pred, padded, key_padding_mask, n_real
        )   # (max_n, N, d_model)

        # Gather EMA targets at masked positions
        full_bf = full_emb.permute(1, 0, 2)                              # (N, S, d)
        pad_exp = padded.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
        true_bf = full_bf.gather(1, pad_exp)                              # (N, max_n, d)
        true_sf = true_bf.permute(1, 0, 2)                               # (max_n, N, d)

        # Mask out padded positions (duplicated last real index)
        max_n    = padded.shape[1]
        valid    = (torch.arange(max_n, device='cuda').unsqueeze(0)
                    < n_real.unsqueeze(1))   # (N, max_n)
        valid_sf = valid.permute(1, 0)       # (max_n, N)

        pred_v = pred_sf[valid_sf].float()
        true_v = true_sf[valid_sf].float()

        if self.cfg.target_norm:
            pred_v = F.normalize(pred_v, dim=-1)
            true_v = F.normalize(true_v, dim=-1)

        return F.smooth_l1_loss(pred_v, true_v.detach(), beta=0.5)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(self, data, compute_reg=True):
        """
        Returns:
            total_loss:       scalar tensor
            token_loss:       float or None (for logging)
            block_loss:       float or None (for logging)
        """
        commands = data['command'].cuda()
        args     = data['args'].cuda()

        # Masking — returns mask and per-sequence regime labels
        target_mask, regime_per_seq = self.masker(commands)

        if target_mask.sum() == 0:
            zero = torch.tensor(0.0, device='cuda', requires_grad=True)
            return zero, None, None

        # Online encoder — sees masked sequence
        ctx_emb, _ = self.encoder(commands, args, target_mask=target_mask)

        # EMA target encoder — sees full unmasked sequence, no grad
        with torch.no_grad():
            full_emb, _ = self.ema(commands, args)

        # Key padding mask for predictor attention
        commands_sf      = commands.permute(1, 0)
        key_padding_mask = _get_key_padding_mask(commands_sf, seq_dim=0)

        # Zero out masked positions in context before predictor
        # Ensures predictor can ONLY attend to true context positions
        mask_sf      = target_mask.permute(1, 0).unsqueeze(-1)
        ctx_for_pred = ctx_emb.masked_fill(mask_sf, 0.0)

        # ── Per-regime prediction loss ────────────────────────────────────
        # Split batch by masking regime for separate loss tracking.
        # Both regimes use identical loss — split is for logging only.
        token_idx = [i for i, r in enumerate(regime_per_seq) if r == 'token']
        block_idx = [i for i, r in enumerate(regime_per_seq) if r == 'block']

        losses        = []
        token_loss_v  = None
        block_loss_v  = None

        for idx_list, regime_name in [(token_idx, 'token'), (block_idx, 'block')]:
            if not idx_list:
                continue
            idx_t = torch.tensor(idx_list, device='cuda')
            loss_r = self._prediction_loss(
                ctx_for_pred[:, idx_t, :],
                full_emb[:, idx_t, :],
                target_mask[idx_t, :],
                key_padding_mask[idx_t, :],
            )
            losses.append(loss_r)
            if regime_name == 'token':
                token_loss_v = loss_r.item()
            else:
                block_loss_v = loss_r.item()

        if not losses:
            zero = torch.tensor(0.0, device='cuda', requires_grad=True)
            return zero, None, None

        pred_loss = torch.stack(losses).mean()

        # ── Variance regularisation ───────────────────────────────────────
        if compute_reg and self.cfg.vicreg_lambda_v > 0.0:
            var_loss   = self._variance_loss(ctx_emb, commands, target_mask)
            total_loss = pred_loss + var_loss
        else:
            total_loss = pred_loss

        return total_loss, token_loss_v, block_loss_v

    # ── Train / val loops ─────────────────────────────────────────────────────

    def _train_epoch(self, loader, epoch, total_steps):
        self.encoder.train()
        self.predictor.train()

        losses       = []
        token_losses = []
        block_losses = []
        pbar = tqdm(loader, desc=f'Ep{epoch:04d}', leave=False)

        for data in pbar:
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, tok_l, blk_l = self._forward(data, compute_reg=True)

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()),
                self.cfg.grad_clip
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self._update_ema(total_steps)

            self.global_step += 1
            losses.append(loss.item())
            if tok_l is not None:
                token_losses.append(tok_l)
            if blk_l is not None:
                block_losses.append(blk_l)

            pbar.set_postfix(loss=f'{loss.item():.5f}')

        mean_loss  = float(np.mean(losses))
        mean_token = float(np.mean(token_losses)) if token_losses else None
        mean_block = float(np.mean(block_losses)) if block_losses else None
        return mean_loss, mean_token, mean_block

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.encoder.eval()
        self.predictor.eval()
        losses = []
        for data in tqdm(loader, desc='Val', leave=False):
            loss, _, _ = self._forward(data, compute_reg=False)
            losses.append(loss.item())
        return float(np.mean(losses))

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_ckpt(self, epoch, tag=None):
        name = tag or f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':       epoch,
            'global_step': self.global_step,
            'encoder':     self.encoder.state_dict(),
            'ema_encoder': self.ema.encoder.state_dict(),
            'predictor':   self.predictor.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
        }, path)
        return path

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.ema.encoder.load_state_dict(ckpt['ema_encoder'])
        self.predictor.load_state_dict(ckpt['predictor'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self):
        cfg          = self.cfg
        train_loader = get_dataloader('train',      cfg)
        val_loader   = get_dataloader('validation', cfg)

        total_steps = cfg.nr_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        start_epoch = 0
        if cfg.cont:
            ckpt_path   = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'[Trainer] Resumed from epoch {start_epoch}')

        log_path = os.path.join(cfg.log_dir, 'losses.txt')
        print(f'[Trainer] Total steps: {total_steps} | '
              f'Steps/epoch: {len(train_loader)}')

        for epoch in range(start_epoch, cfg.nr_epochs):

            # ── Train ─────────────────────────────────────────────────────
            train_loss, token_loss, block_loss = self._train_epoch(
                train_loader, epoch, total_steps
            )

            # ── Val ───────────────────────────────────────────────────────
            val_loss = None
            if epoch % cfg.val_frequency == 0:
                val_loss = self._val_epoch(val_loader)

            # ── EMA momentum at this epoch (for logging) ──────────────────
            mom_now = self._get_ema_momentum(total_steps)
            lr_now  = self.optimizer.param_groups[0]['lr']

            # ── Logging ───────────────────────────────────────────────────
            val_str   = f'{val_loss:.5f}' if val_loss  is not None else '-------'
            tok_str   = f'{token_loss:.5f}' if token_loss is not None else '---'
            blk_str   = f'{block_loss:.5f}' if block_loss is not None else '---'

            msg = (f'ep={epoch:04d}  '
                   f'train={train_loss:.5f}  '
                   f'val={val_str}  '
                   f'tok={tok_str}  '
                   f'blk={blk_str}  '
                   f'lr={lr_now:.2e}  '
                   f'mom={mom_now:.4f}  '
                   f'step={self.global_step}')
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

            # ── Collapse monitoring (every 10 epochs after burn-in) ───────
            if epoch % 10 == 0:
                rank = self.monitor.compute_rank(
                    self.ema.encoder,
                    self.monitor_cmds,
                    self.monitor_args
                )
                self._max_rank_seen = max(self._max_rank_seen, rank)
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
                    print(f'*** COLLAPSE WARNING ep={epoch} '
                          f'rank={rank:.3f} < 50% of max={self._max_rank_seen:.3f} ***')

            # ── Checkpoints ───────────────────────────────────────────────
            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)
            self.save_ckpt(epoch, tag='latest')

        print('[Trainer] Training complete.')