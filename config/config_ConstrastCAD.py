# %%writefile /content/DeepCAD_prashant/config/configContrastCAD.py
import os
import json
import shutil
import argparse
from cadlib.macro import *


class ConfigContrastCAD(object):
    def __init__(self, phase):
        self.is_train = (phase == 'train')
        self._set_defaults()
        args = self._parse()

        print("---- ContrastCAD Configuration ----")
        for k, v in sorted(args.__dict__.items()):
            print(f"  {k:35s} {v}")
            setattr(self, k, v)

        self.exp_dir   = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir   = os.path.join(self.exp_dir,  'log')
        self.model_dir = os.path.join(self.exp_dir,  'model')

        if self.is_train and not self.cont and os.path.exists(self.exp_dir):
            print(f"[Config] Overwriting {self.exp_dir}")
            shutil.rmtree(self.exp_dir)

        for d in [self.log_dir, self.model_dir]:
            os.makedirs(d, exist_ok=True)

        if self.gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids)

        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def _set_defaults(self):
        # Encoder — identical to all other 3 encoders
        self.args_dim        = ARGS_DIM
        self.n_args          = N_ARGS
        self.n_commands      = len(ALL_COMMANDS)
        self.n_layers        = 4
        self.n_layers_decode = 4
        self.n_heads         = 8
        self.dim_feedforward = 512
        self.d_model         = 256
        self.dropout         = 0.1
        self.dim_z           = 256
        self.use_group_emb   = False   # CRITICAL
        self.max_n_ext       = MAX_N_EXT
        self.max_n_loops     = MAX_N_LOOPS
        self.max_n_curves    = MAX_N_CURVES
        self.max_num_groups  = 30
        self.max_total_len   = MAX_TOTAL_LEN

        # Contrastive specific
        self.latent_dropout  = 0.1     # dropout rate for two views
        self.n_phead_layers  = 2       # projection head depth
        self.phead_type      = 'multi' # 'legacy' or 'multi'
        self.temperature     = 0.07
        self.cl_loss         = 'infonce'

        self.loss_weights = {
            "loss_cmd_weight":  1.0,
            "loss_args_weight": 2.0,
            "loss_cl_weight":   1.0,
        }

    def _parse(self):
        p = argparse.ArgumentParser()
        p.add_argument('--proj_dir',          type=str,   default='proj_log')
        p.add_argument('--data_root',         type=str,   default='/content/deepcad_data')
        p.add_argument('--exp_name',          type=str,   default='contrastcad_paper')
        p.add_argument('-g', '--gpu_ids',     type=str,   default='0')
        p.add_argument('--dedup_ids_path',    type=str,
                       default='/content/deepcad_data/train_dedup_ids.json')
        p.add_argument('--batch_size',        type=int,   default=256)
        p.add_argument('--num_workers',       type=int,   default=4)
        p.add_argument('--nr_epochs',         type=int,   default=400)
        p.add_argument('--lr',                type=float, default=1e-3)
        p.add_argument('--grad_clip',         type=float, default=1.0)
        p.add_argument('--warmup_step',       type=int,   default=2000)
        p.add_argument('--save_frequency',    type=int,   default=10)
        p.add_argument('--val_frequency',     type=int,   default=10)
        p.add_argument('--augment',           action='store_true', default=True)
        p.add_argument('--dataset_augment_type', type=str, default='rre')
        p.add_argument('--dataset_augment_prob', type=float, default=0.5)
        p.add_argument('--continue', dest='cont', action='store_true')
        p.add_argument('--ckpt',              type=str,   default='latest')
        p.add_argument('--temperature',       type=float, default=0.07)
        p.add_argument('--latent_dropout',    type=float, default=0.1)
        p.add_argument('--cl_loss',           type=str,   default='infonce')
        args, _ = p.parse_known_args()
        return args