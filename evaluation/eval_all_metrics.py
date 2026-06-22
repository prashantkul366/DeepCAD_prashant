"""
eval_all_metrics.py  — Run on Colab (A100)
===========================================
Comprehensive evaluation suite for all H-CAD-JEPA models.

Metrics computed:
  1. Standard retrieval       — mAP@10, R@1 (all models)
  2. Per-class mAP@10         — 4 complexity classes (all models)
  3. Shape-signature retrieval— fine-grained mAP@10, immune to group-emb leak
  4. Clustering quality       — Silhouette Coefficient, NMI, SSE (all models)
  5. K-NN classification      — shape signature labels, K=1,5,10 (all models)
  6. Linear probe             — fine-grained property prediction (all models)
  7. Embedding geometry       — alignment, uniformity (all models)

Usage:
  python eval_all_metrics.py
  Results saved to {PROJ}/eval_results/comprehensive_results.json
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import argparse
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import (average_precision_score,
                             silhouette_score,
                             normalized_mutual_info_score)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

sys.path.insert(0, '/content/DeepCAD_prashant')
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader
from cadlib.macro import EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX

# ── CONFIG ────────────────────────────────────────────────────
PROJ       = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT  = '/content/deepcad_data'
SPLIT_JSON = '/content/deepcad_data/train_val_test_split.json'
LABELS_NPY = '/content/test_labels.npy'
OUT_DIR    = f'{PROJ}/eval_results'
os.makedirs(OUT_DIR, exist_ok=True)

# All models to evaluate
MODELS = {
    # name: (ckpt_path, use_group_emb, key)
    'DeepCAD AE': (
        None,  # uses pre-saved embeddings
        True, 'z'
    ),
    'ContrastCAD+RRE': (
        None,  # uses pre-saved h5
        True, 'test_zs'
    ),
    'VICReg-only': (
        f'{PROJ}/hcadjepa_vicreg_only/model/latest.pt',
        True, 'encoder'
    ),
    'MAE-on-CAD': (
        f'{PROJ}/hcadjepa_mae/model/latest.pt',
        True, 'ema_encoder'
    ),
    'data2vec': (
        f'{PROJ}/hcadjepa_data2vec/model/latest.pt',
        True, 'ema_encoder'
    ),
    'H-CAD-JEPA ep420': (
        f'{PROJ}/hcadjepa_hierarchical/model/ckpt_ep0420.pt',
        True, 'ema_encoder'
    ),
    'H-CAD-JEPA+Jitter': (
        f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt',
        True, 'ema_encoder'
    ),
    'H-CAD-JEPA nogrp': (
        f'{PROJ}/hcadjepa_nogrp/model/latest.pt',
        False, 'ema_encoder'
    ),
    'H-CAD-JEPA+Jitter nogrp': (
        f'{PROJ}/hcadjepa_jitter_nogrp/model/latest.pt',
        False, 'ema_encoder'
    ),
}
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════

def load_test_metadata():
    """Load test IDs, labels, raw sequences."""
    with open(SPLIT_JSON) as f:
        test_ids = json.load(f)['test']
    labels = np.load(LABELS_NPY)
    return test_ids, labels


def load_raw_sequences(test_ids, n=None):
    """Load raw h5 vectors for test sequences."""
    ids = test_ids[:n] if n else test_ids
    vecs = {}
    for seq_id in tqdm(ids, desc='Loading sequences', leave=False):
        h5_path = os.path.join(DATA_ROOT, 'cad_vec', seq_id + '.h5')
        with h5py.File(h5_path, 'r') as f:
            vecs[seq_id] = f['vec'][:]
    return vecs


def make_eval_cfg(use_group_emb=True):
    return argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.1, use_group_emb=use_group_emb,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=256, num_workers=4,
        data_root=DATA_ROOT, use_cls=False,
    )


def extract_embeddings(ckpt_path, use_group_emb=True, key='ema_encoder'):
    """Extract test embeddings from a JEPA checkpoint."""
    cfg     = make_eval_cfg(use_group_emb)
    loader  = get_dataloader('test', cfg, shuffle=False)
    encoder = JEPAEncoder(cfg).cuda()
    encoder.eval()
    raw = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    encoder.load_state_dict(raw[key])
    epoch = raw.get('epoch', '?')
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            z = encoder.get_pooled_embedding(
                batch['command'].cuda(), batch['args'].cuda())
            all_embs.append(z.cpu().numpy())
    return normalize(np.concatenate(all_embs)), epoch


def load_embeddings_for_model(name, config):
    """Load or extract embeddings for a named model."""
    ckpt_path, use_group_emb, key = config

    # Pre-saved embeddings
    if name == 'DeepCAD AE':
        emb_path = f'{PROJ}/cadjjepa_block/embeddings/deepcad_ae_embeddings.npy'
        if not os.path.exists(emb_path):
            print(f"  [SKIP] {name}: embeddings not found at {emb_path}")
            return None, None
        return normalize(np.load(emb_path)), 'pretrained'

    if name == 'ContrastCAD+RRE':
        h5_path = f'{PROJ}/contrastcad_rre_embeddings.h5'
        if not os.path.exists(h5_path):
            print(f"  [SKIP] {name}: embeddings not found at {h5_path}")
            return None, None
        with h5py.File(h5_path, 'r') as f:
            return normalize(f['test_zs'][:].astype(np.float32)), 'pretrained'

    # Extract from checkpoint
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {name}: checkpoint not found at {ckpt_path}")
        return None, None

    print(f"  Extracting {name}...")
    return extract_embeddings(ckpt_path, use_group_emb, key)


# ══════════════════════════════════════════════════════════════
#  Shape signatures
# ══════════════════════════════════════════════════════════════

def compute_shape_signatures(test_ids):
    """
    Fine-grained shape label = (n_ext, n_line, n_arc, n_circle).
    Creates 100-200 unique classes vs 4 coarse classes.
    Completely immune to group-embedding leakage.
    """
    sigs = []
    for seq_id in tqdm(test_ids, desc='Computing signatures', leave=False):
        h5_path = os.path.join(DATA_ROOT, 'cad_vec', seq_id + '.h5')
        with h5py.File(h5_path, 'r') as f:
            cmds = f['vec'][:, 0]
        sig = (
            int((cmds == EXT_IDX).sum()),
            int((cmds == LINE_IDX).sum()),
            int((cmds == ARC_IDX).sum()),
            int((cmds == CIRCLE_IDX).sum()),
        )
        sigs.append(sig)
    return sigs


def signatures_to_labels(sigs):
    """Convert signature tuples to integer class labels."""
    unique = sorted(set(sigs))
    sig2int = {s: i for i, s in enumerate(unique)}
    return np.array([sig2int[s] for s in sigs]), sig2int


# ══════════════════════════════════════════════════════════════
#  Metric 1: Standard retrieval
# ══════════════════════════════════════════════════════════════

def compute_retrieval(embs, labels, k=10):
    """mAP@k and R@1."""
    N = len(embs)
    aps, r1s = [], []
    for i in range(N):
        sims = embs @ embs[i]
        sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        r1s.append(int(labels[ranked[0]] == labels[i]))
        top_k    = ranked[:k]
        rel      = (labels[top_k] == labels[i]).astype(int)
        if rel.sum() > 0:
            aps.append(average_precision_score(rel, sims[top_k]))
    return float(np.mean(aps)) * 100, float(np.mean(r1s)) * 100


# ══════════════════════════════════════════════════════════════
#  Metric 2: Per-class mAP@10
# ══════════════════════════════════════════════════════════════

def compute_perclass_map(embs, labels, k=10):
    """mAP@10 for each of the 4 complexity classes."""
    per_class = {}
    for cls in sorted(np.unique(labels)):
        cls_mask = (labels == cls)
        cls_aps  = []
        for i in np.where(cls_mask)[0]:
            sims = embs @ embs[i]; sims[i] = -1
            ranked = np.argsort(sims)[::-1]
            top_k  = ranked[:k]
            rel    = (labels[top_k] == labels[i]).astype(int)
            if rel.sum() > 0:
                cls_aps.append(average_precision_score(rel, sims[top_k]))
        per_class[int(cls)] = float(np.mean(cls_aps)) * 100 if cls_aps else 0.0
    return per_class


# ══════════════════════════════════════════════════════════════
#  Metric 3: Shape-signature retrieval
# ══════════════════════════════════════════════════════════════

def compute_signature_retrieval(embs, sig_labels, k=10):
    """
    Fine-grained mAP@10 using shape signatures as ground truth.
    Random baseline ~1-2% (vs 56% for 4-class).
    """
    N = len(embs)
    aps = []
    for i in range(N):
        sims = embs @ embs[i]; sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        top_k  = ranked[:k]
        rel    = (sig_labels[top_k] == sig_labels[i]).astype(int)
        if rel.sum() > 0:
            aps.append(average_precision_score(rel, sims[top_k]))
    return float(np.mean(aps)) * 100 if aps else 0.0


# ══════════════════════════════════════════════════════════════
#  Metric 4: Clustering quality
# ══════════════════════════════════════════════════════════════

def compute_clustering(embs, labels, n_clusters=4):
    """
    K-Means clustering → Silhouette Coefficient, NMI, SSE.
    SC closer to 1 = better clustering.
    NMI: how well clusters align with ground-truth classes.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(embs)
    sc  = float(silhouette_score(embs, cluster_labels, sample_size=2000))
    nmi = float(normalized_mutual_info_score(labels, cluster_labels))
    sse = float(km.inertia_)
    return {'SC': sc, 'NMI': nmi, 'SSE': sse}


# ══════════════════════════════════════════════════════════════
#  Metric 5: K-NN classification
# ══════════════════════════════════════════════════════════════

def compute_knn(embs, sig_labels, coarse_labels, Ks=(1, 5, 10)):
    """
    K-NN classification accuracy with both label types.
    Split: first 80% as gallery, last 20% as query.
    """
    N     = len(embs)
    split = int(0.8 * N)
    train_embs, test_embs   = embs[:split],       embs[split:]
    train_sig,  test_sig    = sig_labels[:split],  sig_labels[split:]
    train_coarse, test_coarse = coarse_labels[:split], coarse_labels[split:]

    results = {}
    for k in Ks:
        knn_sig = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn_sig.fit(train_embs, train_sig)
        acc_sig = float(knn_sig.score(test_embs, test_sig)) * 100

        knn_c = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn_c.fit(train_embs, train_coarse)
        acc_c = float(knn_c.score(test_embs, test_coarse)) * 100

        results[f'KNN-{k}_sig']    = acc_sig
        results[f'KNN-{k}_coarse'] = acc_c

    return results


# ══════════════════════════════════════════════════════════════
#  Metric 6: Linear probe
# ══════════════════════════════════════════════════════════════

def compute_linear_probe(embs, labels, sig_labels):
    """
    Linear probing for:
      - 4-class operation count (coarse)
      - Shape signature class (fine-grained, 100+ classes)
      - Has-arc binary (does sequence contain any ARC command?)
    Uses logistic regression with L2 regularization.
    """
    results = {}
    N = len(embs)

    # 4-class coarse
    lr_coarse = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(lr_coarse, embs, labels, cv=5, scoring='accuracy')
    results['LP_coarse_4class'] = float(scores.mean()) * 100

    # Fine-grained shape signature
    # Only classes with >=5 samples (for CV)
    unique, counts = np.unique(sig_labels, return_counts=True)
    valid_classes  = unique[counts >= 5]
    mask = np.isin(sig_labels, valid_classes)
    if mask.sum() > 100:
        lr_sig = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores_sig = cross_val_score(
            lr_sig, embs[mask], sig_labels[mask], cv=5, scoring='accuracy'
        )
        results['LP_sig_finegrained'] = float(scores_sig.mean()) * 100

    return results


# ══════════════════════════════════════════════════════════════
#  Metric 7: Embedding geometry
# ══════════════════════════════════════════════════════════════

def compute_embedding_geometry(embs, labels):
    """
    Alignment: mean pairwise distance within same class (lower = tighter).
    Uniformity: log mean pairwise Gaussian kernel (lower = more uniform).
    """
    # Alignment: within-class mean distance
    align_vals = []
    for cls in np.unique(labels):
        cls_embs = embs[labels == cls]
        if len(cls_embs) < 2:
            continue
        idx = np.random.choice(len(cls_embs), min(200, len(cls_embs)), replace=False)
        e = cls_embs[idx]
        dists = 1 - e @ e.T  # cosine distance
        triu  = dists[np.triu_indices(len(e), k=1)]
        align_vals.extend(triu.tolist())
    alignment = float(np.mean(align_vals))

    # Uniformity on unit sphere
    idx = np.random.choice(len(embs), min(2000, len(embs)), replace=False)
    e   = embs[idx]
    G   = e @ e.T
    uniformity = float(np.log(np.exp(2 * G).mean()))

    return {'alignment': alignment, 'uniformity': uniformity}


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("H-CAD-JEPA Comprehensive Evaluation Suite")
    print("=" * 60)

    # Load metadata
    test_ids, labels = load_test_metadata()
    print(f"Test sequences: {len(test_ids)}")

    # Compute shape signatures (once, shared across all models)
    print("\nComputing shape signatures...")
    sigs     = compute_shape_signatures(test_ids)
    sig_labels, sig2int = signatures_to_labels(sigs)
    n_sig_classes = len(sig2int)
    print(f"  Unique shape signatures: {n_sig_classes} (vs 4 coarse classes)")
    print(f"  Random baseline (sig retrieval): ~{100/n_sig_classes:.1f}%")
    print(f"  Random baseline (4-class retrieval): ~{100/4:.1f}%")

    # Save signature info
    np.save(f'{OUT_DIR}/sig_labels.npy', sig_labels)
    with open(f'{OUT_DIR}/sig2int.json', 'w') as f:
        json.dump({str(k): v for k, v in sig2int.items()}, f, indent=2)

    all_results = {}

    # Evaluate each model
    for name, config in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"Model: {name}")

        embs, epoch = load_embeddings_for_model(name, config)
        if embs is None:
            continue

        print(f"  Embeddings shape: {embs.shape}  epoch={epoch}")
        results = {'epoch': str(epoch)}

        # 1. Standard retrieval
        mAP, R1 = compute_retrieval(embs, labels)
        results['mAP@10']  = round(mAP, 2)
        results['R@1']     = round(R1, 2)
        print(f"  mAP@10={mAP:.2f}%  R@1={R1:.2f}%")

        # 2. Per-class mAP
        pc = compute_perclass_map(embs, labels)
        results['per_class_mAP'] = {str(k): round(v, 2) for k, v in pc.items()}
        print(f"  Per-class: 1-op={pc[0]:.2f}  2-op={pc[1]:.2f}  "
              f"3-5op={pc[2]:.2f}  6+op={pc[3]:.2f}")

        # 3. Shape-signature retrieval
        sig_mAP = compute_signature_retrieval(embs, sig_labels)
        results['sig_mAP@10'] = round(sig_mAP, 2)
        print(f"  Sig-mAP@10={sig_mAP:.2f}% (random~{100/n_sig_classes:.1f}%)")

        # 4. Clustering
        clust = compute_clustering(embs, labels)
        results['clustering'] = {k: round(v, 4) for k, v in clust.items()}
        print(f"  SC={clust['SC']:.4f}  NMI={clust['NMI']:.4f}  "
              f"SSE={clust['SSE']:.1f}")

        # 5. K-NN classification
        knn = compute_knn(embs, sig_labels, labels)
        results['knn'] = {k: round(v, 2) for k, v in knn.items()}
        print(f"  K-NN(1) sig={knn['KNN-1_sig']:.2f}%  "
              f"coarse={knn['KNN-1_coarse']:.2f}%")

        # 6. Linear probe
        lp = compute_linear_probe(embs, labels, sig_labels)
        results['linear_probe'] = {k: round(v, 2) for k, v in lp.items()}
        print(f"  LP coarse={lp.get('LP_coarse_4class', 0):.2f}%  "
              f"sig={lp.get('LP_sig_finegrained', 0):.2f}%")

        # 7. Embedding geometry
        geo = compute_embedding_geometry(embs, labels)
        results['geometry'] = {k: round(v, 4) for k, v in geo.items()}
        print(f"  Align={geo['alignment']:.4f}  "
              f"Uniform={geo['uniformity']:.4f}")

        all_results[name] = results

        # Save embeddings for NN reconstruction
        emb_save = os.path.join(OUT_DIR, f"{name.replace(' ', '_').replace('+', 'plus')}_embs.npy")
        np.save(emb_save, embs)

    # Save all results
    out_path = f'{OUT_DIR}/comprehensive_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved → {out_path}")

    # Print summary table
    print("\n── SUMMARY TABLE ──────────────────────────────────────────")
    print(f"{'Method':<25} {'mAP@10':>8} {'6+op':>8} {'SigMap':>8} "
          f"{'SC':>7} {'NMI':>7} {'KNN-1':>7}")
    print("─" * 72)
    for name, r in all_results.items():
        pc  = r.get('per_class_mAP', {})
        knn = r.get('knn', {})
        clu = r.get('clustering', {})
        print(f"{name:<25} "
              f"{r.get('mAP@10', 0):>8.2f} "
              f"{pc.get('3', 0):>8.2f} "
              f"{r.get('sig_mAP@10', 0):>8.2f} "
              f"{clu.get('SC', 0):>7.4f} "
              f"{clu.get('NMI', 0):>7.4f} "
              f"{knn.get('KNN-1_sig', 0):>7.2f}")


if __name__ == '__main__':
    main()