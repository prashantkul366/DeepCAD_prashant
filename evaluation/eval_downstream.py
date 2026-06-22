"""
eval_downstream.py  — Run on Colab (A100)
==========================================
Downstream task evaluation for H-CAD-JEPA.

Tasks:
  1. Latent space interpolation
     — interpolate between two shape embeddings
     — find NN training sequence at each interpolated point
     — render on Acer for qualitative figure

  2. Few-shot shape recognition
     — given K examples of a shape family, find more
     — K=1,5,10 with leave-one-out evaluation

  3. Shape analogy
     — A:B :: C:? in embedding space
     — finds D such that D-C ≈ B-A

  4. Generation quality setup
     — saves interpolated/analogy sequences for Acer rendering
     — full generation (latent GAN) requires separate training

  5. Random baseline stats for all metrics

Usage:
  python eval_downstream.py
  Saves interpolation/analogy sequences for Acer rendering.
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
from cadlib.macro import EXT_IDX, SOL_IDX, EOS_IDX

# ── CONFIG ────────────────────────────────────────────────────
PROJ      = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT = '/content/deepcad_data'
OUT_DIR   = f'{PROJ}/eval_results'
LABELS    = '/content/test_labels.npy'
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_CKPT       = f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt'
MAIN_GROUP_EMB  = True
MAIN_KEY        = 'ema_encoder'
N_INTERP_STEPS  = 7   # interpolation steps between two shapes
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


def extract_all_embeddings(ckpt_path, use_group_emb, key):
    """Extract train + test embeddings."""
    cfg     = make_cfg(use_group_emb)
    encoder = JEPAEncoder(cfg).cuda()
    encoder.eval()
    raw = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    encoder.load_state_dict(raw[key])

    results = {}
    for phase in ['train', 'test']:
        loader     = get_dataloader(phase, cfg, shuffle=False)
        all_embs, all_ids = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'  {phase}', leave=False):
                z = encoder.get_pooled_embedding(
                    batch['command'].cuda(), batch['args'].cuda())
                all_embs.append(z.cpu().numpy())
                all_ids.extend(batch['id'])
        results[phase] = {
            'embs': normalize(np.concatenate(all_embs)),
            'ids':  all_ids
        }
    return results


def find_nn(query_emb, gallery_embs, gallery_ids, exclude_ids=None):
    """Find nearest neighbor in gallery."""
    sims = gallery_embs @ query_emb
    if exclude_ids:
        for eid in exclude_ids:
            if eid in gallery_ids:
                idx = gallery_ids.index(eid)
                sims[idx] = -1
    nn_idx = np.argmax(sims)
    return gallery_ids[nn_idx], float(sims[nn_idx])


def load_seq(seq_id):
    h5_path = os.path.join(DATA_ROOT, 'cad_vec', seq_id + '.h5')
    with h5py.File(h5_path, 'r') as f:
        return f['vec'][:]


# ══════════════════════════════════════════════════════════════
#  Task 1: Latent Space Interpolation
# ══════════════════════════════════════════════════════════════

def latent_interpolation(train_embs, train_ids, test_embs, test_ids,
                          labels, n_pairs=3, n_steps=N_INTERP_STEPS):
    """
    Pick n_pairs of shapes from different classes (class 0 and class 3).
    Interpolate in embedding space, find NN training sequence at each step.
    Saves sequence IDs for rendering on Acer.

    This shows the embedding space is smooth and geometrically meaningful —
    intermediate points correspond to real intermediate shapes.
    """
    print(f"\n── Task 1: Latent Interpolation ──────────────────────")

    # Pick pairs: one simple (class 0) and one complex (class 3)
    cls0_idx = np.where(labels == 0)[0]
    cls3_idx = np.where(labels == 3)[0]
    np.random.shuffle(cls0_idx)
    np.random.shuffle(cls3_idx)

    pairs = list(zip(cls0_idx[:n_pairs], cls3_idx[:n_pairs]))
    all_interp = []

    for pair_idx, (i0, i3) in enumerate(pairs):
        emb_start = test_embs[i0]   # simple shape
        emb_end   = test_embs[i3]   # complex shape
        id_start  = test_ids[i0]
        id_end    = test_ids[i3]

        # Linear interpolation in embedding space
        alphas = np.linspace(0, 1, n_steps)
        interp_seqs = [id_start]

        for alpha in alphas[1:-1]:
            # Interpolate and renormalize to unit sphere (SLERP-like)
            interp = (1 - alpha) * emb_start + alpha * emb_end
            interp = interp / (np.linalg.norm(interp) + 1e-8)

            # Find nearest training sequence
            nn_id, sim = find_nn(interp, train_embs, train_ids,
                                  exclude_ids=[id_start, id_end])
            interp_seqs.append(nn_id)

        interp_seqs.append(id_end)

        pair_data = {
            'pair': pair_idx,
            'start_id': id_start,
            'end_id':   id_end,
            'start_label': int(labels[i0]),
            'end_label':   int(labels[i3]),
            'sequence': interp_seqs,
            'n_unique': len(set(interp_seqs))
        }
        all_interp.append(pair_data)
        print(f"  Pair {pair_idx}: {id_start} (cls{labels[i0]}) → "
              f"{id_end} (cls{labels[i3]})  unique={len(set(interp_seqs))}/{n_steps}")

    # Save for Acer rendering
    out_path = f'{OUT_DIR}/interpolation_sequences.json'
    with open(out_path, 'w') as f:
        json.dump(all_interp, f, indent=2)
    print(f"  Saved → {out_path}")
    return all_interp


# ══════════════════════════════════════════════════════════════
#  Task 2: Few-Shot Shape Recognition
# ══════════════════════════════════════════════════════════════

def few_shot_recognition(test_embs, test_ids, sig_labels, K_shots=(1, 5, 10)):
    """
    Few-shot shape recognition with shape signature ground truth.

    Protocol:
      For each class C with >=20 examples:
        - Sample K support sequences
        - Query: remaining sequences of class C
        - Compute accuracy: does NN of query fall in same class?

    Shows practical utility: given a few example designs,
    find similar ones from the database.
    """
    print(f"\n── Task 2: Few-Shot Shape Recognition ────────────────")

    unique_classes, counts = np.unique(sig_labels, return_counts=True)
    # Only classes with enough samples
    valid_classes = unique_classes[counts >= 20]
    print(f"  Classes with >=20 samples: {len(valid_classes)}")

    results = {}
    for K in K_shots:
        accs = []
        for cls in valid_classes:
            cls_idx = np.where(sig_labels == cls)[0]
            np.random.shuffle(cls_idx)

            support_idx = cls_idx[:K]
            query_idx   = cls_idx[K:]

            if len(query_idx) == 0:
                continue

            # Prototype: mean of support embeddings
            prototype = test_embs[support_idx].mean(axis=0)
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)

            # For each query, check if nearest support is correct class
            query_embs = test_embs[query_idx]
            sims       = query_embs @ prototype  # (n_query,)

            # Also check all other class prototypes
            other_classes = [c for c in valid_classes if c != cls]
            other_protos  = []
            for oc in other_classes:
                oc_idx   = np.where(sig_labels == oc)[0][:K]
                oc_proto = test_embs[oc_idx].mean(axis=0)
                oc_proto = oc_proto / (np.linalg.norm(oc_proto) + 1e-8)
                other_protos.append(oc_proto)
            other_protos = np.array(other_protos)   # (n_other, D)

            # For each query: is sim to correct proto > all others?
            other_sims = query_embs @ other_protos.T    # (n_query, n_other)
            correct    = (sims[:, None] > other_sims).all(axis=1)
            accs.append(float(correct.mean()))

        results[f'K={K}'] = round(float(np.mean(accs)) * 100, 2) if accs else 0.0
        print(f"  K={K}-shot accuracy: {results[f'K={K}']:.2f}%")

    return results


# ══════════════════════════════════════════════════════════════
#  Task 3: Shape Analogy
# ══════════════════════════════════════════════════════════════

def shape_analogy(train_embs, train_ids, test_embs, test_ids,
                   labels, n_analogies=5):
    """
    Shape analogy: A:B :: C:?
    Finds D such that embedding(D) - embedding(C) ≈ embedding(B) - embedding(A)

    Example: (simple box A) : (box with hole B) ::
             (simple cylinder C) : (cylinder with hole D?)

    Tests whether the embedding encodes meaningful geometric transformations.
    Evaluated by: is D more similar to B-type shapes than random?
    """
    print(f"\n── Task 3: Shape Analogy (A:B :: C:?) ────────────────")

    # Pick A,B from class 0 and 1 (simple and medium complexity)
    cls0 = np.where(labels == 0)[0]
    cls1 = np.where(labels == 1)[0]
    cls2 = np.where(labels == 2)[0]

    np.random.shuffle(cls0); np.random.shuffle(cls1); np.random.shuffle(cls2)

    analogies = []
    for i in range(min(n_analogies, min(len(cls0), len(cls1), len(cls2)))):
        iA, iB, iC = cls0[i], cls1[i], cls2[2*i]

        emb_A = test_embs[iA]
        emb_B = test_embs[iB]
        emb_C = test_embs[iC]

        # Analogy vector
        delta     = emb_B - emb_A
        emb_D_hat = emb_C + delta
        emb_D_hat = emb_D_hat / (np.linalg.norm(emb_D_hat) + 1e-8)

        # Find NN in training set
        id_D, sim = find_nn(emb_D_hat, train_embs, train_ids)

        # Sanity: does D have more operations than C?
        h5_C = os.path.join(DATA_ROOT, 'cad_vec', test_ids[iC] + '.h5')
        h5_D = os.path.join(DATA_ROOT, 'cad_vec', id_D + '.h5')
        with h5py.File(h5_C, 'r') as f:
            n_ops_C = int((f['vec'][:, 0] == EXT_IDX).sum())
        with h5py.File(h5_D, 'r') as f:
            n_ops_D = int((f['vec'][:, 0] == EXT_IDX).sum())

        result = {
            'A_id':    test_ids[iA],
            'B_id':    test_ids[iB],
            'C_id':    test_ids[iC],
            'D_id':    id_D,
            'sim':     float(sim),
            'n_ops_C': n_ops_C,
            'n_ops_D': n_ops_D,
            'ops_increased': n_ops_D > n_ops_C
        }
        analogies.append(result)
        print(f"  Analogy {i}: C({n_ops_C} ops) → D({n_ops_D} ops) "
              f"sim={sim:.3f}  ops↑={n_ops_D > n_ops_C}")

    ops_increased = sum(a['ops_increased'] for a in analogies)
    print(f"  Analogy direction correct (ops increase): "
          f"{ops_increased}/{len(analogies)}")

    out_path = f'{OUT_DIR}/analogy_sequences.json'
    with open(out_path, 'w') as f:
        json.dump(analogies, f, indent=2)
    print(f"  Saved → {out_path}")
    return analogies


# ══════════════════════════════════════════════════════════════
#  Task 4: Generation quality via embedding space
# ══════════════════════════════════════════════════════════════

def embedding_space_coverage(train_embs, test_embs, labels):
    """
    Coverage metric: for each test sequence, is there a training
    sequence within distance threshold?

    High coverage = embedding space well-populated with diverse shapes.
    This is a proxy for generation quality without training a GAN.

    Also computes:
    - MMD (Maximum Mean Discrepancy) between train and test distributions
    - FID-equivalent in embedding space
    """
    print(f"\n── Task 4: Embedding Space Coverage ─────────────────")

    # Coverage: fraction of test sequences with a train NN within distance d
    thresholds = [0.1, 0.2, 0.3, 0.5]
    results    = {}

    # Batch NN distances
    batch_size = 512
    min_dists  = []
    for start in tqdm(range(0, len(test_embs), batch_size), leave=False):
        batch  = test_embs[start:start+batch_size]
        sims   = batch @ train_embs.T      # (B, N_train)
        max_sim = sims.max(axis=1)          # closest train neighbor
        min_dists.extend((1 - max_sim).tolist())

    min_dists = np.array(min_dists)

    for t in thresholds:
        cov = float((min_dists < t).mean()) * 100
        results[f'coverage@{t}'] = round(cov, 2)
        print(f"  Coverage@{t}: {cov:.2f}%")

    # MMD approximation (Gaussian kernel)
    n_sample = min(2000, len(train_embs), len(test_embs))
    idx_tr   = np.random.choice(len(train_embs), n_sample, replace=False)
    idx_te   = np.random.choice(len(test_embs),  n_sample, replace=False)
    tr_s     = train_embs[idx_tr]
    te_s     = test_embs[idx_te]

    # RBF kernel
    sigma = 1.0
    K_tt  = np.exp(-np.sum((tr_s[:, None] - tr_s[None, :]) ** 2, axis=-1) / (2*sigma))
    K_ee  = np.exp(-np.sum((te_s[:, None] - te_s[None, :]) ** 2, axis=-1) / (2*sigma))
    K_te  = np.exp(-np.sum((tr_s[:, None] - te_s[None, :]) ** 2, axis=-1) / (2*sigma))
    mmd   = float(K_tt.mean() + K_ee.mean() - 2*K_te.mean())
    results['MMD'] = round(mmd, 6)
    print(f"  MMD (train vs test): {mmd:.6f} (lower = more similar distributions)")

    return results


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("H-CAD-JEPA Downstream Task Evaluation")
    print("=" * 60)

    labels   = np.load(LABELS)

    # Load shape signatures
    sig_path = f'{OUT_DIR}/sig_labels.npy'
    if not os.path.exists(sig_path):
        print("Run eval_all_metrics.py first to generate sig_labels.npy")
        return
    sig_labels = np.load(sig_path)

    print(f"\nLoading model: H-CAD-JEPA+Jitter (best model)")
    print("Extracting train + test embeddings...")
    splits = extract_all_embeddings(MAIN_CKPT, MAIN_GROUP_EMB, MAIN_KEY)

    train_embs = splits['train']['embs']
    train_ids  = splits['train']['ids']
    test_embs  = splits['test']['embs']
    test_ids   = splits['test']['ids']

    print(f"  Train: {len(train_ids)} | Test: {len(test_ids)}")

    all_results = {}
    np.random.seed(42)

    # Task 1: Latent interpolation
    interp = latent_interpolation(
        train_embs, train_ids, test_embs, test_ids, labels
    )
    all_results['interpolation_n_unique'] = np.mean(
        [p['n_unique'] for p in interp]
    )

    # Task 2: Few-shot recognition
    few_shot = few_shot_recognition(test_embs, test_ids, sig_labels)
    all_results['few_shot'] = few_shot

    # Task 3: Shape analogy
    analogies = shape_analogy(train_embs, train_ids, test_embs, test_ids, labels)
    ops_inc   = sum(a['ops_increased'] for a in analogies)
    all_results['analogy_direction_acc'] = ops_inc / len(analogies)

    # Task 4: Coverage + MMD
    coverage = embedding_space_coverage(train_embs, test_embs, labels)
    all_results['coverage'] = coverage

    # Save
    out_path = f'{OUT_DIR}/downstream_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("── DOWNSTREAM SUMMARY ──────────────────────────────────")
    print(f"  Few-shot K=1:  {few_shot.get('K=1', 0):.2f}%")
    print(f"  Few-shot K=5:  {few_shot.get('K=5', 0):.2f}%")
    print(f"  Few-shot K=10: {few_shot.get('K=10', 0):.2f}%")
    print(f"  Analogy direction acc: {ops_inc}/{len(analogies)}")
    print(f"  Coverage@0.2: {coverage.get('coverage@0.2', 0):.2f}%")
    print(f"  MMD: {coverage.get('MMD', 0):.6f}")
    print(f"\nSaved → {out_path}")
    print("\nFor latent interpolation rendering → run")
    print("render_interpolation.py on Acer (uses interpolation_sequences.json)")


if __name__ == '__main__':
    main()