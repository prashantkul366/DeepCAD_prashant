import os
import json
import shutil
import argparse
from cadlib.macro import (
    ARGS_DIM, N_ARGS, ALL_COMMANDS,
    MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN
)


def _ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class ConfigJEPA:
    def __init__(self, phase):
        self.is_train = (phase == 'train')
        self._set_defaults()
        args = self._parse()

        print("---- JEPA Configuration ----")
        for k, v in sorted(args.__dict__.items()):
            print(f"  {k:35s} {v}")
            setattr(self, k, v)

        # Derived paths
        self.exp_dir   = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir   = os.path.join(self.exp_dir,  'log')
        self.model_dir = os.path.join(self.exp_dir,  'model')

        if self.is_train and not self.cont and os.path.exists(self.exp_dir):
            print(f"[Config] Overwriting existing experiment at {self.exp_dir}")
            shutil.rmtree(self.exp_dir)

        _ensure_dirs([self.log_dir, self.model_dir])

        if self.gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids)

        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def _set_defaults(self):
        # ── Sequence format ───────────────────────────────────────────────
        self.args_dim        = ARGS_DIM           # 256
        self.n_args          = N_ARGS             # 16
        self.n_commands      = len(ALL_COMMANDS)  # 6
        self.max_n_ext       = MAX_N_EXT
        self.max_n_loops     = MAX_N_LOOPS
        self.max_n_curves    = MAX_N_CURVES
        self.max_total_len   = MAX_TOTAL_LEN      # 60
        self.max_num_groups  = 30

        # ── Encoder architecture ──────────────────────────────────────────
        self.d_model         = 256
        self.n_layers        = 4
        self.n_heads         = 8
        self.dim_feedforward = 512
        self.dropout         = 0.1

        # Group embedding MUST be False for paper experiments.
        # Group embedding encodes block-count (the evaluation label) —
        # including it inflates all downstream metrics artificially.
        self.use_group_emb   = False
        self.use_cls         = False   # CLS token vs mean pool for downstream

        # ── EMA ───────────────────────────────────────────────────────────
        # Momentum schedule handled in trainer — four phases.
        # ema_decay is the final momentum value (phase 4).
        # Warmup ramps from 0.990 → ema_decay over ema_warmup_steps steps.
        self.ema_decay        = 0.999
        self.ema_warmup_steps = 2000   # steps to reach full momentum

        # ── Predictor ─────────────────────────────────────────────────────
        # Narrower than encoder (pred_dim < d_model) — prevents shortcut.
        # Shallow (pred_depth=2) — must not be powerful enough to cheat.
        self.pred_dim         = 128
        self.pred_depth       = 2
        self.pred_heads       = 4
        self.pred_ffn_dim     = 256    # 2× pred_dim
        self.max_seq_len      = MAX_TOTAL_LEN

        # ── Loss ──────────────────────────────────────────────────────────
        # Smooth L1 beta=0.5: more L1-like in [0,1] range,
        # less sensitive to large errors in early training.
        # target_norm=False: raw embeddings, not normalized.
        # Normalized targets on homogeneous CAD data create collapsed fixed point.
        self.target_norm      = False

        # ── Masking ───────────────────────────────────────────────────────
        # anchor_block: JEPA masker — block 0 always kept, mask from blocks[1:]
        # token:        MAE masker  — random curve tokens
        self.masking_strategy = 'anchor_block'
        self.mask_ratio        = 0.40  # block masking ratio
        self.mask_ratio_token  = 0.50  # token masking ratio (single-block fallback)

        # ── Variance regularisation ───────────────────────────────────────
        # Variance term only (no covariance) — minimal collapse prevention.
        # Add covariance (vicreg_lambda_c > 0) only if collapse detected.
        # Weight 0.05: small enough not to dominate JEPA loss (~0.3-1.5).
        self.vicreg_lambda_v  = 0.05
        self.vicreg_lambda_c  = 0.0

        # ── Augmentation ──────────────────────────────────────────────────
        # augment:      block-swap between sequences (off for clean JEPA)
        # jitter_aug:   ±jitter_strength noise on curve arg values
        # translate_aug: global sketch translation ±translate_strength units
        #   Attacks x=128 center-bias (37% of Line x values in EDA).
        self.augment           = False
        self.jitter_aug        = True
        self.jitter_strength   = 2
        self.translate_aug     = True
        self.translate_strength = 15

    def _parse(self):
        p = argparse.ArgumentParser()

        # ── Paths ─────────────────────────────────────────────────────────
        p.add_argument('--proj_dir',    type=str, default='proj_log')
        p.add_argument('--data_root',   type=str, default='data')
        p.add_argument('--exp_name',    type=str, default='cad_jepa')
        p.add_argument('--dedup_ids_path', type=str,
                       default='data/train_dedup_ids.json',
                       help='Path to HNC-CAD deduplicated training IDs')

        # ── Hardware ──────────────────────────────────────────────────────
        p.add_argument('-g', '--gpu_ids',   type=str, default='0')
        p.add_argument('--num_workers',     type=int, default=4)

        # ── Training ──────────────────────────────────────────────────────
        p.add_argument('--batch_size',  type=int,   default=256)
        p.add_argument('--nr_epochs',   type=int,   default=400)
        p.add_argument('--lr',          type=float, default=1.5e-4)
        p.add_argument('--grad_clip',   type=float, default=1.0)
        p.add_argument('--warmup_step', type=int,   default=2000)
        p.add_argument('--seed',        type=int,   default=42)

        # ── Checkpointing ─────────────────────────────────────────────────
        p.add_argument('--save_frequency', type=int, default=10)
        p.add_argument('--val_frequency',  type=int, default=5)
        p.add_argument('--continue',  dest='cont', action='store_true')
        p.add_argument('--ckpt',      type=str, default='latest')

        # ── Architecture ──────────────────────────────────────────────────
        p.add_argument('--d_model',         type=int, default=256)
        p.add_argument('--n_layers',        type=int, default=4)
        p.add_argument('--n_heads',         type=int, default=8)
        p.add_argument('--dim_feedforward', type=int, default=512)
        p.add_argument('--dropout',         type=float, default=0.1)
        p.add_argument('--use_cls',         action='store_true', default=False)

        # ── EMA ───────────────────────────────────────────────────────────
        p.add_argument('--ema_decay',         type=float, default=0.999)
        p.add_argument('--ema_warmup_steps',  type=int,   default=2000)

        # ── Predictor ─────────────────────────────────────────────────────
        p.add_argument('--pred_dim',    type=int, default=128)
        p.add_argument('--pred_depth',  type=int, default=2)
        p.add_argument('--pred_heads',  type=int, default=4)
        p.add_argument('--pred_ffn_dim',type=int, default=256)

        # ── Masking ───────────────────────────────────────────────────────
        p.add_argument('--masking_strategy', type=str, default='anchor_block',
                       choices=['anchor_block', 'token'])
        p.add_argument('--mask_ratio',       type=float, default=0.40)
        p.add_argument('--mask_ratio_token', type=float, default=0.50)

        # ── Loss ──────────────────────────────────────────────────────────
        p.add_argument('--target_norm',     action='store_true', default=False)
        p.add_argument('--vicreg_lambda_v', type=float, default=0.05)
        p.add_argument('--vicreg_lambda_c', type=float, default=0.0)

        # ── Augmentation ──────────────────────────────────────────────────
        p.add_argument('--augment',             action='store_true', default=False)
        p.add_argument('--jitter_aug',          action='store_true', default=True)
        p.add_argument('--jitter_strength',     type=int,   default=2)
        p.add_argument('--translate_aug',       action='store_true', default=True)
        p.add_argument('--translate_strength',  type=int,   default=15)

        return p.parse_args()