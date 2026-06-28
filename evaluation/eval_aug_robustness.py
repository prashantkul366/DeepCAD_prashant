"""
eval_aug_robustness.py
──────────────────────
Jitter augmentation robustness analysis.

For each encoder:
  - Take N_SAMPLES multi-block test sequences
  - For each sequence: embed(x) once, then embed(jitter_k(x)) N_AUG times
  - Compute cosine_sim(embed(x), embed(jitter_k(x))) for each k
  - Report mean ± std per encoder

Jitter implementation is IDENTICAL to training (cad_dataset_new.py):
  - Applied to LINE/ARC/CIRCLE tokens only
  - ±JITTER_STR (=2) on all arg columns
  - Command column (col 0) is protected
  - Clip to [0, 255]

Hypothesis:
  JEPA >> MAE > AE in cosine similarity under jitter.
  JEPA was trained to ignore parameter noise (jitter is in training aug).
  AE was trained to reconstruct exact parameters — sensitive to any change.

Usage in Colab:
  %run /content/DeepCAD_prashant/evaluation/eval_aug_robustness.py

Outputs:
  {DRIVE}/robustness_eval/robustness_results.json
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

sys.path.insert(0, '/content/DeepCAD_prashant')

# ── Paths ──────────────────────────────────────────────────────────────────────
DRIVE     = '/content/drive/MyDrive/cadjepa_data'
DATA_ROOT = '/content/deepcad_data'
OUT_DIR   = f'{DRIVE}/robustness_eval'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_SAMPLES     = 500   # sequences from test set (multi-block only)
N_AUG         = 10    # jitter variants per sequence
JITTER_STR    = 2     # must match configJEPA_new.py jitter_strength
TRANSLATE_STR = 15    # must match configJEPA_new.py translate_strength
MAX_TOTAL_LEN = 60
BATCH_SIZE    = 64    # for encoder forward passes

from cadlib.macro import (
    EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX,
    EOS_VEC
)


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation functions (identical to cad_dataset_new.py)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_jitter(cad_vec_np, strength=JITTER_STR):
    """
    cad_vec_np: (MAX_TOTAL_LEN, 17) int64 — padded sequence
    Applies ±strength jitter to curve token arg columns.
    Command column (col 0) is never touched.
    Returns modified copy.
    """
    CURVE_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
    cad_vec = cad_vec_np.copy()
    is_curve = np.array([int(c) in CURVE_CMDS for c in cad_vec[:, 0]])
    if is_curve.any():
        s = strength
        jitter = np.random.randint(-s, s + 1, size=cad_vec[is_curve].shape)
        jitter[:, 0] = 0  # protect command column
        cad_vec[is_curve] = np.clip(
            cad_vec[is_curve].astype(np.int32) + jitter, 0, 255
        ).astype(cad_vec_np.dtype)
    return cad_vec


def apply_translate(cad_vec_np, strength=TRANSLATE_STR):
    """
    Global x,y translation of all curve tokens.
    Shifts col 1 (x) and col 2 (y) by a single random offset.
    Returns modified copy.
    """
    CURVE_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
    cad_vec = cad_vec_np.copy().astype(np.int32)
    is_curve = np.array([int(c) in CURVE_CMDS for c in cad_vec[:, 0]])
    if is_curve.any():
        dx = np.random.randint(-strength, strength + 1)
        dy = np.random.randint(-strength, strength + 1)
        cad_vec[is_curve, 1] = np.clip(cad_vec[is_curve, 1] + dx, 0, 255)
        cad_vec[is_curve, 2] = np.clip(cad_vec[is_curve, 2] + dy, 0, 255)
    return cad_vec.astype(cad_vec_np.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Test dataset — raw padded sequences, no context split needed
# ═══════════════════════════════════════════════════════════════════════════════

class RawTestDataset(Dataset):
    """
    Loads raw padded sequences from the test split.
    Filters to multi-block only (n_ops >= 2) for consistency with completion eval.
    Stops after N_SAMPLES sequences.
    """
    def __init__(self, data_root, n_samples=N_SAMPLES):
        split_path = os.path.join(data_root, 'train_val_test_split.json')
        with open(split_path) as f:
            all_ids = json.load(f)['test']

        raw_data = os.path.join(data_root, 'cad_vec')
        self.samples = []  # list of (n_ops, padded_vec_np)

        for data_id in all_ids:
            h5_path = os.path.join(raw_data, data_id + '.h5')
            try:
                with h5py.File(h5_path, 'r') as fp:
                    vec = fp['vec'][:]               # (raw_len, 17)
            except Exception:
                continue

            n_ops = int((vec[:, 0] == EXT_IDX).sum())
            if n_ops < 2:
                continue  # skip single-block

            pad = MAX_TOTAL_LEN - vec.shape[0]
            padded = np.concatenate(
                [vec, EOS_VEC[np.newaxis].repeat(pad, axis=0)], axis=0
            )  # (60, 17)

            self.samples.append((n_ops, padded))
            if len(self.samples) >= n_samples:
                break

        print(f"[RawTestDataset] Loaded {len(self.samples)} multi-block sequences "
              f"(n_ops >= 2)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n_ops, padded = self.samples[idx]
        return {
            'vec':  torch.tensor(padded, dtype=torch.long),  # (60, 17)
            'n_ops': n_ops,
        }

    def get_all_vecs_np(self):
        """Return (N, 60, 17) numpy array and n_ops list for batched ops."""
        vecs   = np.stack([s[1] for s in self.samples], axis=0)
        n_ops  = [s[0] for s in self.samples]
        return vecs, n_ops


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_ae_encoder():
    from model.autoencoder import CADTransformer
    from config.configAE_paper import ConfigAEPaper
    sys.argv = ['eval']
    cfg = ConfigAEPaper('train')
    net = CADTransformer(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/ae_paper/latest.pt',
                      map_location='cuda', weights_only=False)
    net.load_state_dict(ckpt['net'])
    return net.encoder.eval()


def load_mae_encoder():
    from model.jepa_encoder import JEPAEncoder
    from config.configJEPA_new import ConfigJEPA
    sys.argv = ['eval', '--exp_name', 'eval']
    cfg = ConfigJEPA('train')
    enc = JEPAEncoder(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/mae_run1/latest.pt',
                      map_location='cuda', weights_only=False)
    enc.load_state_dict(ckpt['encoder'])
    return enc.eval()


def load_jepa_encoder():
    from model.jepa_encoder import JEPAEncoder
    from config.configJEPA_new import ConfigJEPA
    sys.argv = ['eval', '--exp_name', 'eval']
    cfg = ConfigJEPA('train')
    enc = JEPAEncoder(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/jepa_run4/latest.pt',
                      map_location='cuda', weights_only=False)
    enc.load_state_dict(ckpt['ema_encoder'])
    return enc.eval()


def load_contrastcad_encoder():
    """Run after ContrastCAD finishes training (~ep400)."""
    from trainer.trainerContrastCAD import TrainerContrastCAD
    from config.configContrastCAD import ConfigContrastCAD
    sys.argv = ['eval']
    cfg = ConfigContrastCAD('train')
    trainer = TrainerContrastCAD(cfg)
    ckpt = torch.load(f'{DRIVE}/contrastcad_paper/latest.pt',
                      map_location='cuda', weights_only=False)
    trainer.net.load_state_dict(ckpt['net'])
    trainer.proj.load_state_dict(ckpt['proj'])
    return trainer.net.encoder.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# Batched robustness computation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_batch(encoder, vecs_np, batch_size=BATCH_SIZE):
    """
    vecs_np: (N, 60, 17) numpy int64
    Returns: (N, 256) L2-normalized float32 tensor on CPU
    """
    N = vecs_np.shape[0]
    all_emb = []
    for i in range(0, N, batch_size):
        batch = vecs_np[i:i + batch_size]
        cmd  = torch.tensor(batch[:, :, 0], dtype=torch.long).cuda()   # (B, 60)
        args = torch.tensor(batch[:, :, 1:], dtype=torch.long).cuda()  # (B, 60, 16)
        emb = encoder.get_pooled_embedding(cmd, args)                   # (B, 256)
        emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)  # (N, 256)


@torch.no_grad()
def compute_robustness(encoder, vecs_np, n_ops_list, aug_fn, aug_name, n_aug=N_AUG):
    """
    vecs_np:    (N, 60, 17) int64 — original padded sequences
    n_ops_list: list of int — n_ops per sequence
    aug_fn:     function(vec_np) -> jittered_vec_np
    aug_name:   str label for printing
    n_aug:      number of augmentation rounds

    Returns: list of dicts per sequence:
      { n_ops, mean_sim, std_sim }
    Similarity is cosine sim between original and augmented embedding.
    """
    N = vecs_np.shape[0]
    print(f"  Computing original embeddings...")
    emb_orig = embed_batch(encoder, vecs_np)  # (N, 256), L2-normalized

    # Accumulate cosine sims across aug rounds: (N, n_aug)
    all_sims = np.zeros((N, n_aug), dtype=np.float32)

    for k in tqdm(range(n_aug), desc=f'  {aug_name} aug rounds'):
        # Apply augmentation independently to each sequence
        vecs_aug = np.stack([aug_fn(vecs_np[i]) for i in range(N)], axis=0)
        emb_aug = embed_batch(encoder, vecs_aug)  # (N, 256)

        # Cosine similarity: dot product of L2-normalized vectors
        sims = (emb_orig * emb_aug).sum(dim=-1).numpy()  # (N,)
        all_sims[:, k] = sims

    # Per-sequence mean and std across aug rounds
    results = []
    for i in range(N):
        results.append({
            'n_ops':    n_ops_list[i],
            'mean_sim': float(all_sims[i].mean()),
            'std_sim':  float(all_sims[i].std()),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Summarization
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_robustness(results):
    GROUPS = [
        ('Overall', 0,   999),
        ('2-3op',   2,   3),
        ('4-5op',   4,   5),
        ('6+op',    6,   999),
    ]
    summary = {}
    for label, lo, hi in GROUPS:
        subset = results if label == 'Overall' else [
            r for r in results if lo <= r['n_ops'] <= hi
        ]
        if not subset:
            summary[label] = {'n': 0, 'mean_sim': 0.0, 'std_sim': 0.0}
            continue
        sims = [r['mean_sim'] for r in subset]
        summary[label] = {
            'n':        len(subset),
            'mean_sim': float(np.mean(sims)),
            'std_sim':  float(np.std(sims)),   # std across sequences
        }
    return summary


def print_robustness_table(all_summaries):
    GROUPS = ['Overall', '2-3op', '4-5op', '6+op']
    header = (f"\n{'Method':<16} {'Group':<10} {'N':>5} "
              f"{'Mean Cos Sim':>14} {'Std':>8}")
    print(header)
    print('─' * 58)
    for method, summary in all_summaries.items():
        for g in GROUPS:
            row = summary[g]
            label = method if g == 'Overall' else ''
            print(f"{label:<16} {g:<10} {row['n']:>5} "
                  f"{row['mean_sim']:>13.4f}  "
                  f"{row['std_sim']:>7.4f}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Load dataset once — reuse across all encoders ──────────────────────────
    print(f"Loading {N_SAMPLES} multi-block test sequences...")
    dataset  = RawTestDataset(DATA_ROOT, n_samples=N_SAMPLES)
    vecs_np, n_ops_list = dataset.get_all_vecs_np()
    print(f"  n_ops distribution: "
          f"2-3op={sum(1 for n in n_ops_list if 2<=n<=3)}, "
          f"4-5op={sum(1 for n in n_ops_list if 4<=n<=5)}, "
          f"6+op={sum(1 for n in n_ops_list if n>=6)}")

    # ── Encoder list ───────────────────────────────────────────────────────────
    ENCODERS = {
        'AE':   load_ae_encoder,
        'MAE':  load_mae_encoder,
        'JEPA': load_jepa_encoder,
        # 'ContrastCAD': load_contrastcad_encoder,  # add when done
    }

    all_summaries = {}
    all_raw       = {}

    for name, load_fn in ENCODERS.items():
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        encoder = load_fn()

        # ── Jitter robustness ──────────────────────────────────────────────────
        jitter_results = compute_robustness(
            encoder, vecs_np, n_ops_list,
            aug_fn=apply_jitter, aug_name='Jitter',
            n_aug=N_AUG
        )
        jitter_summary = summarize_robustness(jitter_results)

        # ── Translation robustness ─────────────────────────────────────────────
        translate_results = compute_robustness(
            encoder, vecs_np, n_ops_list,
            aug_fn=apply_translate, aug_name='Translate',
            n_aug=N_AUG
        )
        translate_summary = summarize_robustness(translate_results)

        all_summaries[name] = {
            'jitter':    jitter_summary,
            'translate': translate_summary,
        }
        all_raw[name] = {
            'jitter':    jitter_results,
            'translate': translate_results,
        }

        print(f"\n  {name}  [Jitter ±{JITTER_STR}]  Overall mean_sim = "
              f"{jitter_summary['Overall']['mean_sim']:.4f}")
        print(f"  {name}  [Translate ±{TRANSLATE_STR}]  Overall mean_sim = "
              f"{translate_summary['Overall']['mean_sim']:.4f}")

        del encoder
        torch.cuda.empty_cache()

    # ── Print tables ───────────────────────────────────────────────────────────
    print("\n\n══ JITTER ROBUSTNESS (±2 quantization units) ══")
    print_robustness_table({m: v['jitter']    for m, v in all_summaries.items()})

    print("══ TRANSLATION ROBUSTNESS (±15 coordinate units) ══")
    print_robustness_table({m: v['translate'] for m, v in all_summaries.items()})

    # ── Save to Drive ──────────────────────────────────────────────────────────
    out_path = f'{OUT_DIR}/robustness_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'summaries': all_summaries,
            'raw':       all_raw,
            'config': {
                'n_samples':    len(dataset),
                'n_aug':        N_AUG,
                'jitter_str':   JITTER_STR,
                'translate_str': TRANSLATE_STR,
            }
        }, f, indent=2)
    print(f"\n[saved] {out_path}")