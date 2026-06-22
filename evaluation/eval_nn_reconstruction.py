"""
eval_nn_reconstruction.py  — Run on Colab (A100)
=================================================
Nearest-Neighbor based reconstruction quality.

Idea: if the embedding space is good, the nearest neighbor
in the TRAINING set for any TEST sequence should be a
structurally similar CAD sequence.

Metrics (matching ContrastCAD Table 4):
  ACC_cmd   — command type accuracy of NN vs query
  ACC_param — parameter accuracy (within tolerance ±3)
  Sequence edit distance — how many tokens differ

Stronger claim: we retrieve a training sequence whose
STRUCTURE matches the query, without any decoder.

Usage:
  python eval_nn_reconstruction.py
  Results saved to {PROJ}/eval_results/nn_reconstruction.json
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import argparse
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.insert(0, '/content/DeepCAD_prashant')
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader
from cadlib.macro import (EXT_IDX, SOL_IDX, EOS_IDX, LINE_IDX,
                           ARC_IDX, CIRCLE_IDX, ALL_COMMANDS,
                           CMD_ARGS_MASK)

# ── CONFIG ────────────────────────────────────────────────────
PROJ      = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT = '/content/deepcad_data'
LABELS    = '/content/test_labels.npy'
OUT_DIR   = f'{PROJ}/eval_results'
os.makedirs(OUT_DIR, exist_ok=True)

TOLERANCE = 3   # parameter tolerance (same as DeepCAD paper)

MODELS = {
    'H-CAD-JEPA+Jitter': (
        f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt',
        True, 'ema_encoder'
    ),
    'H-CAD-JEPA ep420': (
        f'{PROJ}/hcadjepa_hierarchical/model/ckpt_ep0420.pt',
        True, 'ema_encoder'
    ),
    'H-CAD-JEPA+Jitter nogrp': (
        f'{PROJ}/hcadjepa_jitter_nogrp/model/latest.pt',
        False, 'ema_encoder'
    ),
    'MAE-on-CAD': (
        f'{PROJ}/hcadjepa_mae/model/latest.pt',
        True, 'ema_encoder'
    ),
}
# ─────────────────────────────────────────────────────────────


def make_cfg(use_group_emb=True):
    return argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.1, use_group_emb=use_group_emb,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=256, num_workers=4,
        data_root=DATA_ROOT, use_cls=False,
    )


def extract_split_embeddings(ckpt_path, use_group_emb, key, phase):
    """Extract embeddings for train or test split."""
    cfg     = make_cfg(use_group_emb)
    loader  = get_dataloader(phase, cfg, shuffle=False)
    encoder = JEPAEncoder(cfg).cuda()
    encoder.eval()
    raw = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    encoder.load_state_dict(raw[key])

    all_embs, all_ids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  Extracting {phase}', leave=False):
            z = encoder.get_pooled_embedding(
                batch['command'].cuda(), batch['args'].cuda())
            all_embs.append(z.cpu().numpy())
            all_ids.extend(batch['id'])
    return normalize(np.concatenate(all_embs)), all_ids


def load_seq(seq_id):
    """Load raw CAD vector for a sequence."""
    h5_path = os.path.join(DATA_ROOT, 'cad_vec', seq_id + '.h5')
    with h5py.File(h5_path, 'r') as f:
        return f['vec'][:]


def compute_sequence_metrics(query_vec, retrieved_vec):
    """
    Compute ACC_cmd, ACC_param between two CAD sequences.
    Both vectors shape (seq_len, 17): col0=cmd, col1:=args.
    Matches ContrastCAD evaluation protocol.
    """
    # Align to shorter length
    min_len = min(len(query_vec), len(retrieved_vec))
    q = query_vec[:min_len]
    r = retrieved_vec[:min_len]

    q_cmd, r_cmd = q[:, 0].astype(int), r[:, 0].astype(int)
    q_arg, r_arg = q[:, 1:].astype(int), r[:, 1:].astype(int)

    # Command accuracy
    cmd_match = (q_cmd == r_cmd)
    acc_cmd   = float(cmd_match.mean())

    # Parameter accuracy (only where commands match, only valid args)
    args_mask = CMD_ARGS_MASK  # (n_commands, n_args) bool
    param_accs = []
    for i in range(min_len):
        c = q_cmd[i]
        if c in (SOL_IDX, EOS_IDX):
            continue
        if not cmd_match[i]:
            continue
        valid = args_mask[c].astype(bool)
        if not valid.any():
            continue
        tol_match = (np.abs(q_arg[i] - r_arg[i]) <= TOLERANCE)[valid]
        param_accs.extend(tol_match.tolist())

    acc_param = float(np.mean(param_accs)) if param_accs else 0.0

    # Length penalty: if sequences have different lengths
    length_ratio = min_len / max(len(query_vec), len(retrieved_vec))

    return acc_cmd, acc_param, length_ratio


def evaluate_nn_reconstruction(name, ckpt_path, use_group_emb, key):
    """
    For each test sequence:
      1. Find nearest neighbor in training set
      2. Compare sequences: ACC_cmd, ACC_param
    """
    print(f"\n  Extracting train embeddings ({name})...")
    train_embs, train_ids = extract_split_embeddings(
        ckpt_path, use_group_emb, key, 'train'
    )
    print(f"  Train: {len(train_ids)} sequences")

    print(f"  Extracting test embeddings...")
    test_embs, test_ids = extract_split_embeddings(
        ckpt_path, use_group_emb, key, 'test'
    )
    print(f"  Test: {len(test_ids)} sequences")

    # Batch NN search
    acc_cmds, acc_params, length_ratios = [], [], []
    batch_size = 256

    print(f"  Computing NN reconstruction quality...")
    for start in tqdm(range(0, len(test_embs), batch_size), leave=False):
        batch_embs = test_embs[start:start + batch_size]   # (B, D)
        sims       = batch_embs @ train_embs.T             # (B, N_train)
        nn_indices = np.argmax(sims, axis=1)               # (B,)

        for bi, nn_idx in enumerate(nn_indices):
            test_id  = test_ids[start + bi]
            train_id = train_ids[nn_idx]

            q_vec = load_seq(test_id)
            r_vec = load_seq(train_id)

            acc_cmd, acc_param, lr = compute_sequence_metrics(q_vec, r_vec)
            acc_cmds.append(acc_cmd)
            acc_params.append(acc_param)
            length_ratios.append(lr)

    results = {
        'ACC_cmd':       round(float(np.mean(acc_cmds)) * 100, 2),
        'ACC_param':     round(float(np.mean(acc_params)) * 100, 2),
        'length_ratio':  round(float(np.mean(length_ratios)) * 100, 2),
    }

    print(f"  ACC_cmd={results['ACC_cmd']:.2f}%  "
          f"ACC_param={results['ACC_param']:.2f}%  "
          f"length={results['length_ratio']:.2f}%")
    return results


def main():
    print("=" * 60)
    print("NN-Based Reconstruction Quality Evaluation")
    print("=" * 60)
    print("Concept: find nearest training sequence → compare structure")
    print("Matches ContrastCAD Table 4 metrics (ACC_cmd, ACC_param)")

    all_results = {}

    for name, (ckpt_path, use_group_emb, key) in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"Model: {name}")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        results = evaluate_nn_reconstruction(name, ckpt_path, use_group_emb, key)
        all_results[name] = results

    out_path = f'{OUT_DIR}/nn_reconstruction.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("── NN RECONSTRUCTION SUMMARY ──────────────────────────")
    print(f"{'Method':<30} {'ACC_cmd':>10} {'ACC_param':>10}")
    print("─" * 55)
    for name, r in all_results.items():
        print(f"{name:<30} {r['ACC_cmd']:>9.2f}%  {r['ACC_param']:>9.2f}%")
    print(f"\nSaved → {out_path}")
    print("\nNote: For CD-based reconstruction quality, run")
    print("eval_nn_recon_cd.py on Windows Acer (needs pythonocc)")


if __name__ == '__main__':
    main()