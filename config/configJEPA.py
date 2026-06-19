import os, json, shutil, argparse
from utils import ensure_dirs
from cadlib.macro import *


class ConfigJEPA:
    def __init__(self, phase):
        self.is_train = phase == "train"
        self.set_configuration()
        parser, args = self.parse()

        print("---- JEPA Configuration -----")
        for k, v in args.__dict__.items():
            print("{0:25}".format(k), v)
            self.__setattr__(k, v)

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and not args.cont and os.path.exists(self.exp_dir):
            response = input('Experiment exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir   = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        if self.is_train:
            with open(f'{self.exp_dir}/config.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        # ── Inherited from DeepCAD ──────────────────────────
        self.args_dim        = ARGS_DIM          # 256
        self.n_args          = N_ARGS            # 16
        self.n_commands      = len(ALL_COMMANDS) # 6
        self.d_model         = 256
        self.n_layers        = 4
        self.n_heads         = 8
        self.dim_feedforward = 512
        self.dropout         = 0.1
        self.use_group_emb   = True
        self.max_n_ext       = MAX_N_EXT         # 10
        self.max_n_loops     = MAX_N_LOOPS       # 6
        self.max_n_curves    = MAX_N_CURVES      # 15
        self.max_num_groups  = 30
        self.max_total_len   = MAX_TOTAL_LEN     # 60

        # ── JEPA specific ───────────────────────────────────
        self.ema_decay       = 0.996
        self.pred_dim        = 128               # narrow predictor width
        self.pred_depth      = 4
        self.pred_heads      = 4
        self.pred_ffn_dim    = 256
        self.target_norm     = True              # normalize targets before loss
        self.mask_ratio      = 0.40              # fraction to mask
        self.n_mask_groups   = 2                 # blocks per group (group-level)
        self.level_probs     = [0.33, 0.34, 0.33]  # [token, block, group]

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--proj_dir',         type=str,   default="proj_log")
        parser.add_argument('--data_root',         type=str,   default="data")
        parser.add_argument('--exp_name',          type=str,   default="cadjjepa_block")
        parser.add_argument('-g', '--gpu_ids',     type=str,   default='0')
        parser.add_argument('--batch_size',        type=int,   default=256)
        parser.add_argument('--num_workers',       type=int,   default=4)
        parser.add_argument('--nr_epochs',         type=int,   default=300)
        parser.add_argument('--lr',                type=float, default=1.5e-4)
        parser.add_argument('--grad_clip',         type=float, default=1.0)
        parser.add_argument('--warmup_step',       type=int,   default=2000)
        parser.add_argument('--masking_strategy',  type=str,   default='block',
                            choices=['block', 'token', 'group', 'hierarchical'])
        parser.add_argument('--save_frequency',    type=int,   default=10)
        parser.add_argument('--val_frequency',     type=int,   default=5)
        parser.add_argument('--augment',           action='store_true')
        parser.add_argument('--continue',  dest='cont', action='store_true')
        parser.add_argument('--ckpt',              type=str,   default='latest')
        args = parser.parse_args()
        return parser, args