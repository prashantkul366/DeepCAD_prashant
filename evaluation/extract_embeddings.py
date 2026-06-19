# %%writefile /content/DeepCAD_prashant/eval/extract_embeddings.py
"""
Extract frozen JEPA embeddings for all sequences in a dataset split.
Uses the EMA encoder (better downstream performance than online encoder).

Usage:
    python eval/extract_embeddings.py \
        --ckpt /path/to/ckpt_ep0049.pt \
        --data_root /content/deepcad_data \
        --phase test \
        --save_prefix emb_ep49
"""
import os, sys, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader
from cadlib.macro import (
    ARGS_DIM, N_ARGS, ALL_COMMANDS,
    MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES,
    MAX_TOTAL_LEN
)


def make_cfg(data_root, batch_size=256):
    class Cfg:
        args_dim        = ARGS_DIM
        n_args          = N_ARGS
        n_commands      = len(ALL_COMMANDS)
        d_model         = 256
        n_layers        = 4
        n_heads         = 8
        dim_feedforward = 512
        dropout         = 0.0        # no dropout during eval
        use_group_emb   = True
        max_n_ext       = MAX_N_EXT
        max_n_loops     = MAX_N_LOOPS
        max_n_curves    = MAX_N_CURVES
        max_num_groups  = 30
        max_total_len   = MAX_TOTAL_LEN
        augment         = False
        num_workers     = 2
    cfg = Cfg()
    cfg.data_root  = data_root
    cfg.batch_size = batch_size
    return cfg


def extract(ckpt_path, data_root, phase='test',
            batch_size=256, save_prefix='emb'):
    cfg     = make_cfg(data_root, batch_size)
    encoder = JEPAEncoder(cfg).cuda()

    ckpt    = torch.load(ckpt_path, map_location='cuda')
    # Use EMA encoder — it produces better representations
    encoder.load_state_dict(ckpt['ema_encoder'])
    encoder.eval()
    print(f"Loaded EMA encoder from {ckpt_path}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    loader       = get_dataloader(phase, cfg, shuffle=False)
    all_embs, all_ids = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Extracting ({phase})'):
            commands = batch['command'].cuda()
            args     = batch['args'].cuda()
            z        = encoder.get_pooled_embedding(commands, args)  # (N, d_model)
            all_embs.append(z.cpu().float().numpy())
            all_ids.extend(batch['id'])

    embs = np.concatenate(all_embs, axis=0)   # (total, d_model)
    ids  = np.array(all_ids)

    np.save(f'{save_prefix}_embeddings.npy', embs)
    np.save(f'{save_prefix}_ids.npy',        ids)
    print(f'Saved {len(embs)} embeddings → {save_prefix}_embeddings.npy')
    print(f'Embedding shape: {embs.shape}')
    return embs, ids


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         required=True)
    ap.add_argument('--data_root',    default='data')
    ap.add_argument('--phase',        default='test')
    ap.add_argument('--batch_size',   type=int, default=256)
    ap.add_argument('--save_prefix',  default='emb')
    a = ap.parse_args()
    extract(a.ckpt, a.data_root, a.phase, a.batch_size, a.save_prefix)