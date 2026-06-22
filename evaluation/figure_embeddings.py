"""
figure_embeddings.py  — Run on Colab
======================================
Produces:

Fig D: t-SNE comparison (2×4 grid)
  Top row: colored by 4-class label
  Bottom row: colored by shape signature (fine-grained)
  4 models: ContrastCAD+RRE, SkexGen, MAE-on-CAD, H-CAD-JEPA+Jitter

Fig E: PCA analysis
  Left: PC1/PC2 scatter for our model
  Right: PC1 correlation with op count + PC1 distribution by class

Fig F: Layer-wise mAP@10 line plot
  All models across L0→Final
  Shows prediction objective vs retrieval objective gap

Fig G: Cross-class confusion heatmap
  4×4 matrix: rows=query class, cols=retrieved class
  Shows ContrastCAD retrieves simple shapes for complex queries
  Our model stays within class

Fig H: Sensitivity analysis bar chart
  Block-swap vs param-noise ratio comparison
  JEPA+Jitter vs JEPA ep420

Saves to {PROJ}/figures/
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as PCA_sk
from sklearn.metrics import average_precision_score
from tqdm import tqdm

sys.path.insert(0, '/content/DeepCAD_prashant')
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader
from cadlib.macro import EXT_IDX, EOS_IDX

PROJ      = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT = '/content/deepcad_data'
FIGS_DIR  = f'{PROJ}/figures'
LABELS    = '/content/test_labels.npy'
EMB_DIR   = f'{PROJ}/eval_results'
os.makedirs(FIGS_DIR, exist_ok=True)

# Colors for 4 classes
CLASS_COLORS  = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
CLASS_NAMES   = ['1-op', '2-ops', '3-5-ops', '6+-ops']
CLASS_MARKERS = ['o', 's', '^', 'D']


# ── Load pre-saved embeddings ─────────────────────────────────

def load_embeddings():
    """Load all model embeddings from eval_results."""
    labels = np.load(LABELS)

    emb_map = {
        'ContrastCAD+RRE': None,
        'SkexGen':         f'{EMB_DIR}/skexgen_embs.npy',
        'MAE-on-CAD':      f'{EMB_DIR}/H-CAD-JEPA+Jitter_embs.npy',   # saved by eval_all
        'H-CAD-JEPA+Jitter': f'{EMB_DIR}/H-CAD-JEPA+Jitter_embs.npy',
    }

    # Load ContrastCAD from h5
    cc_path = f'{PROJ}/contrastcad_rre_embeddings.h5'
    import h5py
    with h5py.File(cc_path, 'r') as f:
        cc_embs = normalize(f['test_zs'][:].astype(np.float32))

    # Load all from eval_results directory
    result = {}
    result['ContrastCAD+RRE'] = cc_embs

    # H-CAD-JEPA+Jitter — extract fresh (most important model)
    ckpt = torch.load(f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt',
                      map_location='cuda', weights_only=False)
    cfg  = argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.0, use_group_emb=True,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=256, num_workers=4, data_root=DATA_ROOT, use_cls=False,
    )
    enc = JEPAEncoder(cfg).cuda(); enc.eval()
    enc.load_state_dict(ckpt['ema_encoder'])
    loader = get_dataloader('test', cfg, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in loader:
            z = enc.get_pooled_embedding(batch['command'].cuda(), batch['args'].cuda())
            embs.append(z.cpu().numpy())
    result['H-CAD-JEPA+Jitter'] = normalize(np.concatenate(embs))

    # MAE
    ckpt_mae = torch.load(f'{PROJ}/hcadjepa_mae/model/latest.pt',
                           map_location='cuda', weights_only=False)
    enc_mae = JEPAEncoder(cfg).cuda(); enc_mae.eval()
    enc_mae.load_state_dict(ckpt_mae['ema_encoder'])
    embs_mae = []
    with torch.no_grad():
        for batch in loader:
            z = enc_mae.get_pooled_embedding(batch['command'].cuda(), batch['args'].cuda())
            embs_mae.append(z.cpu().numpy())
    result['MAE-on-CAD'] = normalize(np.concatenate(embs_mae))

    # SkexGen
    skex_path = f'{EMB_DIR}/skexgen_embs.npy'
    if os.path.exists(skex_path):
        result['SkexGen'] = np.load(skex_path)
    else:
        result['SkexGen'] = None

    return result, labels


def load_shape_signatures():
    """Load fine-grained shape signatures (1287 classes)."""
    sig_path = f'{EMB_DIR}/sig_labels.npy'
    if os.path.exists(sig_path):
        return np.load(sig_path)
    return None


# ══════════════════════════════════════════════════════════════
#  Figure D: t-SNE Comparison
# ══════════════════════════════════════════════════════════════

def make_tsne_comparison(embeddings, labels, sig_labels):
    """
    2×4 grid: top=class colored, bottom=sig colored
    """
    models = ['ContrastCAD+RRE', 'SkexGen', 'MAE-on-CAD', 'H-CAD-JEPA+Jitter']
    models = [m for m in models if embeddings.get(m) is not None]

    n_models  = len(models)
    n_sample  = 3000
    np.random.seed(42)
    idx = np.random.choice(len(labels), n_sample, replace=False)

    # Pre-compute t-SNE for each model
    print("  Computing t-SNE...")
    tsne_coords = {}
    for mname in tqdm(models, desc='  t-SNE'):
        emb = embeddings[mname][idx]
        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                    random_state=42, n_jobs=-1)
        tsne_coords[mname] = tsne.fit_transform(emb)

    n_rows = 2 if sig_labels is not None else 1
    fig, axes = plt.subplots(n_rows, n_models,
                              figsize=(n_models * 3.5, n_rows * 3.5),
                              facecolor='white')
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for col, mname in enumerate(models):
        coords = tsne_coords[mname]

        # Row 1: 4-class coloring
        ax = axes[0][col]
        for cls in range(4):
            mask = labels[idx] == cls
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                       s=4, alpha=0.6, linewidths=0,
                       marker=CLASS_MARKERS[cls])
        ax.set_title(mname, fontsize=9, fontweight='bold', pad=4)
        ax.set_axis_off()
        if col == 0:
            ax.set_ylabel('4-class labels', fontsize=9,
                          rotation=0, ha='right', va='center', labelpad=80)

        # Row 2: shape signature coloring (fine-grained)
        if sig_labels is not None and n_rows > 1:
            ax2 = axes[1][col]
            sig_sub = sig_labels[idx]
            # Use a continuous colormap since 1287 classes
            ax2.scatter(coords[:, 0], coords[:, 1],
                        c=sig_sub % 256, cmap='tab20',
                        s=4, alpha=0.5, linewidths=0)
            ax2.set_axis_off()
            if col == 0:
                ax2.set_ylabel('Shape signatures\n(1287 classes)',
                               fontsize=9, rotation=0,
                               ha='right', va='center', labelpad=80)

    # Global legend
    patches = [plt.scatter([], [], c=CLASS_COLORS[c], s=25,
                            label=CLASS_NAMES[c], marker=CLASS_MARKERS[c])
               for c in range(4)]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle('t-SNE Embedding Visualization (n=3000 test sequences)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.3)
    out = f'{FIGS_DIR}/fig_tsne.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure E: PCA Analysis
# ══════════════════════════════════════════════════════════════

def make_pca_analysis(emb_jitter, labels):
    """PCA of H-CAD-JEPA+Jitter embeddings."""
    pca = PCA_sk(n_components=10, random_state=42)
    coords = pca.fit_transform(emb_jitter)
    var    = pca.explained_variance_ratio_

    # Correlation of each PC with operation count
    correlations = []
    for pc in range(10):
        r = np.corrcoef(coords[:, pc], labels)[0, 1]
        correlations.append(abs(r))

    fig = plt.figure(figsize=(13, 4.5), facecolor='white')
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Left: PC1/PC2 scatter
    ax1 = fig.add_subplot(gs[0])
    n_sample = 3000
    np.random.seed(42)
    idx = np.random.choice(len(labels), n_sample, replace=False)
    for cls in range(4):
        mask = labels[idx] == cls
        ax1.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                    c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                    s=5, alpha=0.6, linewidths=0, marker=CLASS_MARKERS[cls])
    ax1.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)', fontsize=10)
    ax1.set_title('PCA — H-CAD-JEPA+Jitter', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, markerscale=2, frameon=False)
    ax1.spines[['top', 'right']].set_visible(False)

    # Middle: PC1 distributions per class (violin)
    ax2 = fig.add_subplot(gs[1])
    pc1_by_class = [coords[labels == cls, 0] for cls in range(4)]
    vp = ax2.violinplot(pc1_by_class, positions=range(4),
                        showmedians=True, showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(CLASS_COLORS[i])
        body.set_alpha(0.7)
    vp['cmedians'].set_color('black')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax2.set_ylabel('PC1 value', fontsize=10)
    ax2.set_title(f'PC1 by class\n(ρ={np.corrcoef(coords[:,0], labels)[0,1]:.3f})',
                  fontsize=10, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)

    # Right: |correlation| of each PC with op count
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(range(1, 11), correlations,
            color=['#E74C3C' if i == 0 else '#85C1E9' for i in range(10)],
            edgecolor='white')
    ax3.set_xlabel('Principal Component', fontsize=10)
    ax3.set_ylabel('|Correlation with op count|', fontsize=10)
    ax3.set_title('PC-Operation Count Correlation\n'
                  'PC1 dominates (ρ≈0.50)',
                  fontsize=10, fontweight='bold')
    ax3.set_xticks(range(1, 11))
    ax3.spines[['top', 'right']].set_visible(False)

    plt.suptitle('PCA Analysis — Geometric Complexity as Primary Axis',
                 fontsize=11, fontweight='bold', y=1.02)
    out = f'{FIGS_DIR}/fig_pca.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure F: Layer-Wise mAP@10
# ══════════════════════════════════════════════════════════════

def make_layerwise_plot():
    """Plot pre-computed layer-wise mAP@10 data."""
    # Data from Cell 1 results (already computed)
    layer_data = {
        'Random Init': {
            'vals':  [99.35, 99.33, 99.31, 99.34, 99.36, 99.38],
            'color': '#95A5A6', 'linestyle': '--', 'marker': 'x'
        },
        'No Group Emb (ep299)': {
            'vals':  [85.92, 85.86, 83.02, 80.59, 77.94, 77.85],
            'color': '#E67E22', 'linestyle': ':', 'marker': 's'
        },
        'H-CAD-JEPA ep420': {
            'vals':  [94.87, 93.19, 92.17, 90.40, 89.35, 89.43],
            'color': '#3498DB', 'linestyle': '-', 'marker': 'o'
        },
        'H-CAD-JEPA+Jitter': {
            'vals':  [97.01, 95.51, 93.75, 93.19, 92.13, 92.31],
            'color': '#2ECC71', 'linestyle': '-', 'marker': 'D'
        },
    }

    labels_x = ['L0\n(Embed)', 'L1', 'L2', 'L3', 'L4', 'Final\n+Norm']
    x = np.arange(len(labels_x))

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='white')

    for mname, mdata in layer_data.items():
        ax.plot(x, mdata['vals'],
                color=mdata['color'],
                linestyle=mdata['linestyle'],
                marker=mdata['marker'],
                linewidth=2, markersize=7,
                label=mname)

    # Shade the "prediction objective gap" region
    jepa_vals = layer_data['H-CAD-JEPA+Jitter']['vals']
    ax.fill_between(x, jepa_vals, [jepa_vals[0]] * len(x),
                    alpha=0.07, color='#2ECC71',
                    label='_nolegend_')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_ylabel('mAP@10 (%)', fontsize=11)
    ax.set_ylim(70, 102)
    ax.set_title('Layer-wise Probing: mAP@10 at Each Encoder Layer\n'
                 'L0 (embedding only) achieves highest retrieval — transformer '
                 'optimizes for prediction, not retrieval',
                 fontsize=10, fontweight='bold')

    ax.annotate('Prediction objective\ncreates this gap',
                xy=(2.5, 91.5), xytext=(1.5, 78),
                fontsize=8.5, color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=1.5))

    ax.legend(fontsize=9, frameon=False, loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    out = f'{FIGS_DIR}/fig_layerwise_map.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure G: Cross-Class Confusion Heatmap
# ══════════════════════════════════════════════════════════════

def compute_confusion(embs, labels, k=10):
    """
    For each query, what fraction of top-k retrieved belong to each class?
    Returns (4, 4) matrix: rows=query class, cols=retrieved class
    """
    N = len(embs)
    confusion = np.zeros((4, 4))
    counts    = np.zeros(4)

    for i in range(N):
        sims   = embs @ embs[i]; sims[i] = -1
        top_k  = np.argsort(sims)[::-1][:k]
        qcls   = labels[i]
        counts[qcls] += 1
        for rcls in range(4):
            confusion[qcls, rcls] += (labels[top_k] == rcls).sum() / k

    for c in range(4):
        if counts[c] > 0:
            confusion[c] /= counts[c]
    return confusion


def make_confusion_heatmaps(embeddings, labels):
    """4×4 confusion heatmaps for key models."""
    models    = ['ContrastCAD+RRE', 'SkexGen', 'MAE-on-CAD', 'H-CAD-JEPA+Jitter']
    models    = [m for m in models if embeddings.get(m) is not None]
    n_models  = len(models)

    fig, axes = plt.subplots(1, n_models,
                              figsize=(n_models * 3.2, 3.5),
                              facecolor='white')

    for col, mname in enumerate(models):
        ax = axes[col]
        conf = compute_confusion(embeddings[mname], labels)

        im = ax.imshow(conf, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8, rotation=30)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)

        for r in range(4):
            for c in range(4):
                color = 'white' if conf[r, c] > 0.5 else 'black'
                ax.text(c, r, f'{conf[r, c]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color=color, fontweight='bold' if r == c else 'normal')

        diag = np.diag(conf).mean()
        ax.set_title(f'{mname}\nDiag={diag:.2f}', fontsize=9, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Query class', fontsize=9)
        ax.set_xlabel('Retrieved class', fontsize=9)

    plt.suptitle('Cross-class Retrieval Confusion\n'
                 '(diagonal = correct class, off-diagonal = wrong class retrieved)',
                 fontsize=10, fontweight='bold', y=1.05)
    plt.tight_layout(pad=0.5)
    out = f'{FIGS_DIR}/fig_confusion.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure H: Sensitivity Analysis Bar Chart
# ══════════════════════════════════════════════════════════════

def make_sensitivity_chart():
    """Bar chart of perturbation sensitivity ratios."""
    perturbations = ['Param ±1', 'Param ±5', 'Curve type\nswap', 'Block\nswap', 'Random\nshape']
    
    # From sensitivity analysis (50 seqs, ep420 baseline=param±1, jitter baseline=param±5)
    jepa_dist    = [0.003, 0.013, 0.084, 0.324, 0.769]  # raw distances
    jitter_dist  = [0.000, 0.005, 0.065, 0.283, 0.560]

    # Normalize to param±1 (jepa) and param±5 (jitter) respectively
    jepa_ratio   = [d / jepa_dist[0] if jepa_dist[0] > 0 else 0 for d in jepa_dist]
    jitter_ratio = [d / jitter_dist[1] if jitter_dist[1] > 0 else 0 for d in jitter_dist]
    # Jitter: param±1 is 0 (complete invariance)
    jitter_ratio[0] = 0.0

    x = np.arange(len(perturbations))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')

    # Left: Raw distances
    ax = axes[0]
    ax.bar(x - w/2, jepa_dist,   w, label='H-CAD-JEPA ep420',  color='#3498DB', alpha=0.85)
    ax.bar(x + w/2, jitter_dist, w, label='H-CAD-JEPA+Jitter', color='#2ECC71', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(perturbations, fontsize=9)
    ax.set_ylabel('Mean cosine distance', fontsize=10)
    ax.set_title('Embedding Distance by Perturbation Type', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate jitter's complete invariance
    ax.annotate('Complete\ninvariance\n(d<1e-7)',
                xy=(0 + w/2, 0.001), xytext=(0.5, 0.15),
                fontsize=8, color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71'))

    # Right: Ratios (log scale)
    ax2 = axes[1]
    ax2.bar(x - w/2, jepa_ratio,   w, label='H-CAD-JEPA ep420',  color='#3498DB', alpha=0.85)
    ax2.bar(x + w/2, jitter_ratio, w, label='H-CAD-JEPA+Jitter', color='#2ECC71', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(perturbations, fontsize=9)
    ax2.set_ylabel('Sensitivity ratio (log scale)', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_title('Sensitivity Ratio vs Baseline Perturbation\n'
                  '113× gap: block-swap vs param-noise in JEPA ep420',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines[['top', 'right']].set_visible(False)

    # Annotate 113×
    ax2.annotate('113×', xy=(3 - w/2, 113), fontsize=9, color='#3498DB',
                 ha='center', va='bottom', fontweight='bold')
    ax2.annotate('57×', xy=(3 + w/2, 57), fontsize=9, color='#2ECC71',
                 ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Sensitivity Analysis: What perturbations change the embedding?',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{FIGS_DIR}/fig_sensitivity.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Embedding Visualization Figures")
    print("=" * 60)

    labels     = np.load(LABELS)
    sig_labels = load_shape_signatures()

    print("\nLoading embeddings...")
    embeddings, labels = load_embeddings()
    for name, emb in embeddings.items():
        if emb is not None:
            print(f"  {name}: {emb.shape}")

    # Fig D: t-SNE
    print("\nFig D: t-SNE comparison...")
    make_tsne_comparison(embeddings, labels, sig_labels)

    # Fig E: PCA
    print("\nFig E: PCA analysis...")
    make_pca_analysis(embeddings['H-CAD-JEPA+Jitter'], labels)

    # Fig F: Layer-wise mAP
    print("\nFig F: Layer-wise mAP@10...")
    make_layerwise_plot()

    # Fig G: Confusion heatmaps
    print("\nFig G: Cross-class confusion...")
    make_confusion_heatmaps(embeddings, labels)

    # Fig H: Sensitivity
    print("\nFig H: Sensitivity analysis chart...")
    make_sensitivity_chart()

    print(f"\n✓ All embedding figures saved to {FIGS_DIR}")


if __name__ == '__main__':
    main()