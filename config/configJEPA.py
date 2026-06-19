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
        parser, args = self._parse()

        print("---- JEPA Configuration ----")
        for k, v in sorted(args.__dict__.items()):
            print(f"  {k:30s} {v}")
            setattr(self, k, v)

        self.exp_dir   = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir   = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')

        if self.is_train and not self.cont and os.path.exists(self.exp_dir):
            print(f'Experiment exists at {self.exp_dir} — overwriting.')
            shutil.rmtree(self.exp_dir)

        _ensure_dirs([self.log_dir, self.model_dir])

        if args.gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)

        if self.is_train:
            with open(f'{self.exp_dir}/config.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def _set_defaults(self):
        # ── Inherited sequence format ─────────────────────────
        self.args_dim        = ARGS_DIM
        self.n_args          = N_ARGS
        self.n_commands      = len(ALL_COMMANDS)
        self.d_model         = 256
        self.n_layers        = 4
        self.n_heads         = 8
        self.dim_feedforward = 512
        self.dropout         = 0.1
        self.use_group_emb   = True
        self.max_n_ext       = MAX_N_EXT
        self.max_n_loops     = MAX_N_LOOPS
        self.max_n_curves    = MAX_N_CURVES
        self.max_num_groups  = 30
        self.max_total_len   = MAX_TOTAL_LEN

        # ── JEPA specific ─────────────────────────────────────
        self.ema_decay         = 0.996
        self.ema_warmup_steps  = 5000   # ramp EMA decay over this many steps
        self.pred_dim          = 128
        self.pred_depth        = 4
        self.pred_heads        = 4
        self.pred_ffn_dim      = 256
        self.target_norm       = False  # raw smooth_l1, no unit sphere collapse
        self.mask_ratio        = 0.40   # block and group level
        self.mask_ratio_token  = 0.70   # token level — harder task for single-block seqs
        self.n_mask_groups     = 2
        self.vicreg_lambda     = 1.0    # VICReg variance coefficient

    def _parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--proj_dir',          type=str,   default='proj_log')
        parser.add_argument('--data_root',          type=str,   default='data')
        parser.add_argument('--exp_name',           type=str,   default='cadjjepa_block')
        parser.add_argument('-g', '--gpu_ids',      type=str,   default='0')
        parser.add_argument('--batch_size',         type=int,   default=256)
        parser.add_argument('--num_workers',        type=int,   default=4)
        parser.add_argument('--nr_epochs',          type=int,   default=300)
        parser.add_argument('--lr',                 type=float, default=1.5e-4)
        parser.add_argument('--grad_clip',          type=float, default=1.0)
        parser.add_argument('--warmup_step',        type=int,   default=2000)
        parser.add_argument('--masking_strategy',   type=str,   default='block',
                            choices=['block', 'token', 'group', 'hierarchical'])
        parser.add_argument('--save_frequency',     type=int,   default=10)
        parser.add_argument('--val_frequency',      type=int,   default=5)
        parser.add_argument('--augment',            action='store_true')
        parser.add_argument('--continue',           dest='cont', action='store_true')
        parser.add_argument('--ckpt',               type=str,   default='latest')
        parser.add_argument('--seed',               type=int,   default=42)
        args = parser.parse_args()
        return parser, args