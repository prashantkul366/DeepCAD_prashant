import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.jepa_encoder import JEPAEncoder
from model.ema_target import EMATargetEncoder
from model.predictor import CADJEPAPredictor, HierarchicalPredictor, JEPAPredictor
from model.masker import get_masker
from model.model_utils import _get_key_padding_mask
from model.collapse_monitor import CollapseMonitor
from dataset.cad_dataset import get_dataloader
from cadlib.macro import EOS_IDX


class TrainerJEPA:
    def __init__(self, cfg):
        self.cfg              = cfg
        self.global_step      = 0
        self.is_hierarchical  = (cfg.masking_strategy == 'hierarchical')
        self.vicreg_lambda_v  = getattr(cfg, 'vicreg_lambda_v', 1.0)
        self.vicreg_lambda_c  = getattr(cfg, 'vicreg_lambda_c', 0.04)
        self._build_models()
        self._build_optimizer()
        self.masker           = get_masker(cfg.masking_strategy, cfg)
        self.monitor          = CollapseMonitor(cfg.d_model, rank_threshold=0.70)
        self._max_rank_seen   = 0.0
        self._rank_burn_in    = 20
        self._load_monitor_sample()

    # ──────────────────────────────────────────────────────────
    #  Build
    # ──────────────────────────────────────────────────────────

    def _build_models(self):
        cfg = self.cfg
        self.encoder = JEPAEncoder(cfg).cuda()
        self.ema     = EMATargetEncoder(self.encoder, decay=cfg.ema_decay)
        self.ema.encoder = self.ema.encoder.cuda()

        max_seq_len = getattr(cfg, 'max_seq_len', cfg.max_total_len)
        pred_kw = dict(
            d_model        = cfg.d_model,
            pred_dim       = cfg.pred_dim,
            n_layers       = cfg.pred_depth,
            n_heads        = cfg.pred_heads,
            dim_feedforward= cfg.pred_ffn_dim,
            dropout        = 0.0,       # no dropout in predictor — deterministic
            max_seq_len    = max_seq_len,
        )

        if self.is_hierarchical:
            self.predictor = HierarchicalPredictor(**pred_kw).cuda()
        else:
            self.predictor = CADJEPAPredictor(**pred_kw).cuda()

        n_enc  = sum(p.numel() for p in self.encoder.parameters())
        n_pred = sum(p.numel() for p in self.predictor.parameters())
        print(f"  Encoder params  : {n_enc/1e6:.2f}M")
        print(f"  Predictor params: {n_pred/1e6:.2f}M")

    def _build_optimizer(self):
        """
        Weight decay exclusion — standard transformer training practice.
        1D parameters (bias, norm scale/bias) and embeddings: no decay.
        All other weight matrices: weight_decay=0.05.
        """
        decay, no_decay = [], []
        seen = set()

        for name, param in (list(self.encoder.named_parameters()) +
                             list(self.predictor.named_parameters())):
            if id(param) in seen:
                continue
            seen.add(id(param))

            if (param.ndim <= 1 or       # bias, LayerNorm scale/bias
                'bias'             in name or
                'norm'             in name or
                'embedding'        in name or
                'embed'            in name or
                'position_queries' in name or
                'mask_embedding'   in name):
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

        n_decay    = sum(p.numel() for p in decay)
        n_no_decay = sum(p.numel() for p in no_decay)
        print(f"  Params with decay   : {n_decay/1e6:.2f}M")
        print(f"  Params without decay: {n_no_decay/1e6:.2f}M")

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
    #  EMA — gradual ramp over ema_warmup_steps
    # ──────────────────────────────────────────────────────────

    def _update_ema(self):
        warmup = getattr(self.cfg, 'ema_warmup_steps', 5000)
        t      = min(1.0, self.global_step / warmup)
        decay  = 0.990 + (self.cfg.ema_decay - 0.990) * t
        self.ema.update(self.encoder, decay=decay)

    # ──────────────────────────────────────────────────────────
    #  Monitor sample
    # ──────────────────────────────────────────────────────────

    def _load_monitor_sample(self, n=512):
        class _Cfg:
            pass
        c              = _Cfg()
        c.data_root    = self.cfg.data_root
        c.batch_size   = n
        c.num_workers  = 0
        c.augment      = False
        c.max_n_loops  = self.cfg.max_n_loops
        c.max_n_curves = self.cfg.max_n_curves
        c.max_total_len = self.cfg.max_total_len
        c.max_n_ext    = self.cfg.max_n_ext

        loader = get_dataloader('validation', c, shuffle=True)
        batch  = next(iter(loader))
        self.monitor_cmds = batch['command'][:n]
        self.monitor_args = batch['args'][:n]
        print(f"  Collapse monitor : {self.monitor_cmds.shape[0]} val sequences")

    # ──────────────────────────────────────────────────────────
    #  Masked index extraction
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_masked_indices(target_mask):
        """
        Extract padded masked position indices from a boolean mask.

        target_mask: (N, S) bool — True at masked positions
        Returns:
            padded: (N, max_n) long — indices padded to max_n (last real index repeated)
            n_real: (N,) long       — actual masked count per sequence
        """
        N      = target_mask.shape[0]
        n_real = target_mask.sum(dim=1)              # (N,)
        max_n  = int(n_real.max().item())

        if max_n == 0:
            return None, n_real

        padded = torch.zeros(N, max_n, dtype=torch.long,
                             device=target_mask.device)
        for i in range(N):
            indices = target_mask[i].nonzero(as_tuple=False).squeeze(-1)
            n = indices.shape[0]
            if n > 0:
                padded[i, :n] = indices
                if n < max_n:
                    padded[i, n:] = indices[-1]   # repeat last for padding

        return padded, n_real

    # ──────────────────────────────────────────────────────────
    #  VICReg regularization
    # ──────────────────────────────────────────────────────────

    def _vicreg_loss(self, ctx_emb_sf, commands, target_mask):
        """
        Full VICReg (variance + covariance) on per-token context embeddings.

        Applied to non-masked, non-EOS tokens — the actual context tokens
        with real content (not mask_embedding positions, not padding).

        Variance term: forces each embedding dimension to have std >= 1
                       across the batch of context tokens.
        Covariance term: forces different embedding dimensions to be
                         uncorrelated — critical on homogeneous CAD data
                         to ensure different dims encode different properties.

        ctx_emb_sf:  (S, N, d) seq-first
        commands:    (N, S) batch-first long
        target_mask: (N, S) bool
        """
        is_eos    = (commands.permute(1, 0) == EOS_IDX)  # (S, N)
        is_masked = target_mask.permute(1, 0)            # (S, N)
        valid     = ~is_eos & ~is_masked                 # (S, N)

        z = ctx_emb_sf.float()[valid]                    # (n_valid, d)

        if z.shape[0] < 2:
            return torch.tensor(0.0, device=ctx_emb_sf.device)

        n, d = z.shape
        z    = z - z.mean(dim=0, keepdim=True)           # center per-dimension

        # Variance: each dim should have std >= 1
        std    = torch.sqrt(z.var(dim=0) + 1e-4)
        v_loss = F.relu(1.0 - std).mean()

        # Covariance: off-diagonal elements should be near zero
        cov      = (z.T @ z) / max(n - 1, 1)            # (d, d)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        c_loss   = off_diag.sum() / d

        return self.vicreg_lambda_v * v_loss + self.vicreg_lambda_c * c_loss

    # ──────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────

    def _forward(self, data, compute_vicreg=True):
        commands = data['command'].cuda()
        args     = data['args'].cuda()

        # ── Masking ──────────────────────────────────────────
        if self.is_hierarchical:
            target_mask, level_per_seq = self.masker(commands)
        else:
            target_mask   = self.masker(commands)
            level_per_seq = None

        if target_mask.sum() == 0:
            return torch.tensor(0.0, device='cuda', requires_grad=True)

        # ── Context encoder ──────────────────────────────────
        # Returns (S, N, d_model) seq-first
        ctx_emb = self.encoder(commands, args, target_mask=target_mask)

        # ── EMA target encoder (no grad) ─────────────────────
        with torch.no_grad():
            full_emb = self.ema(commands, args)   # (S, N, d)

        # ── Key padding mask for predictor ───────────────────
        commands_sf      = commands.permute(1, 0)
        key_padding_mask = _get_key_padding_mask(commands_sf, seq_dim=0)  # (N, S)

        # ── Prediction loss ──────────────────────────────────
        if self.is_hierarchical:
            pred_loss = self._hierarchical_loss(
                ctx_emb, full_emb, target_mask, key_padding_mask, level_per_seq
            )
        else:
            pred_loss = self._single_level_loss(
                ctx_emb, full_emb, target_mask, key_padding_mask
            )

        # ── VICReg regularization ────────────────────────────
        if compute_vicreg and (self.vicreg_lambda_v > 0 or
                                self.vicreg_lambda_c > 0):
            vicreg = self._vicreg_loss(ctx_emb, commands, target_mask)
            loss   = pred_loss + vicreg
        else:
            loss = pred_loss

        return loss

    def _single_level_loss(self, ctx_emb, full_emb, target_mask, key_padding_mask):
        """Prediction loss for non-hierarchical maskers."""
        padded, n_real = self._extract_masked_indices(target_mask)

        if padded is None:
            return torch.tensor(0.0, device='cuda', requires_grad=True)

        # Cross-attention predictor: queries at target positions attend to context
        pred_sf = self.predictor(
            ctx_emb, padded, key_padding_mask, n_real
        )  # (max_n, N, d)

        # Gather EMA targets at padded positions
        full_bf  = full_emb.permute(1, 0, 2)          # (N, S, d)
        pad_exp  = padded.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
        true_bf  = full_bf.gather(1, pad_exp)          # (N, max_n, d)
        true_sf  = true_bf.permute(1, 0, 2)            # (max_n, N, d)

        # Valid positions only (exclude padded)
        max_n    = padded.shape[1]
        valid    = (torch.arange(max_n, device='cuda').unsqueeze(0)
                    < n_real.unsqueeze(1))              # (N, max_n)
        valid_sf = valid.permute(1, 0)                  # (max_n, N)

        pred_v = pred_sf[valid_sf].float()
        true_v = true_sf[valid_sf].float()

        if self.cfg.target_norm:
            pred_v = F.normalize(pred_v, dim=-1)
            true_v = F.normalize(true_v, dim=-1)

        return F.smooth_l1_loss(pred_v, true_v.detach())

    def _hierarchical_loss(self, ctx_emb, full_emb, target_mask,
                            key_padding_mask, level_per_seq):
        """Prediction loss for hierarchical masker — routes by level."""
        losses = []

        for level in ['token', 'block', 'group']:
            idx = [i for i, l in enumerate(level_per_seq) if l == level]
            if not idx:
                continue

            idx_t   = torch.tensor(idx, device='cuda')
            ctx_lv  = ctx_emb[:, idx_t, :]        # (S, n_lv, d)
            full_lv = full_emb[:, idx_t, :]       # (S, n_lv, d)
            mask_lv = target_mask[idx_t, :]       # (n_lv, S)
            kpm_lv  = key_padding_mask[idx_t, :]  # (n_lv, S)

            padded_lv, n_real_lv = self._extract_masked_indices(mask_lv)
            if padded_lv is None:
                continue

            pred_sf = self.predictor(
                ctx_lv, level, padded_lv, kpm_lv, n_real_lv
            )  # (max_n, n_lv, d)

            # Gather EMA targets
            full_bf = full_lv.permute(1, 0, 2)    # (n_lv, S, d)
            pad_exp = padded_lv.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
            true_bf = full_bf.gather(1, pad_exp)   # (n_lv, max_n, d)
            true_sf = true_bf.permute(1, 0, 2)     # (max_n, n_lv, d)

            max_n    = padded_lv.shape[1]
            valid    = (torch.arange(max_n, device='cuda').unsqueeze(0)
                        < n_real_lv.unsqueeze(1))   # (n_lv, max_n)
            valid_sf = valid.permute(1, 0)           # (max_n, n_lv)

            pred_v = pred_sf[valid_sf].float()
            true_v = true_sf[valid_sf].float()

            if self.cfg.target_norm:
                pred_v = F.normalize(pred_v, dim=-1)
                true_v = F.normalize(true_v, dim=-1)

            if pred_v.shape[0] > 0:
                losses.append(F.smooth_l1_loss(pred_v, true_v.detach()))

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
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self._forward(data, compute_vicreg=True)
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
        """Validation: prediction loss only — no VICReg."""
        self.encoder.eval()
        self.predictor.eval()
        losses = []
        for data in tqdm(loader, desc='Val', leave=False):
            losses.append(self._forward(data, compute_vicreg=False).item())
        return float(np.mean(losses))

    # ──────────────────────────────────────────────────────────
    #  Checkpoint
    # ──────────────────────────────────────────────────────────

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

        total_steps = cfg.nr_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        start_epoch = 0
        if cfg.cont:
            ckpt_path   = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'Resumed from epoch {start_epoch}')

        log_path = os.path.join(cfg.log_dir, 'losses.txt')

        for epoch in range(start_epoch, cfg.nr_epochs):
            train_loss = self._train_epoch(train_loader, epoch)

            val_loss = 0.0
            if epoch % cfg.val_frequency == 0:
                val_loss = self._val_epoch(val_loader)

            lr_now = self.optimizer.param_groups[0]['lr']
            msg    = (f'ep={epoch:04d}  train={train_loss:.5f}  '
                      f'val={val_loss:.5f}  lr={lr_now:.2e}  '
                      f'step={self.global_step}')
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

            # ── Rank monitoring ────────────────────────────────
            rank = self.monitor.compute_rank(
                self.ema.encoder, self.monitor_cmds, self.monitor_args
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
                print(f'  *** COLLAPSE WARNING ep={epoch} '
                      f'rank={rank:.3f} < 50% of max={self._max_rank_seen:.3f} ***')

            # ── Checkpoints ────────────────────────────────────
            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)
            self.save_ckpt(epoch, tag='latest')

        print('Training complete.')