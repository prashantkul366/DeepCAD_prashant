"""
figure_explainability.py  — Run on Colab
=========================================
Produces 3 explainability figures:

Fig A: Token saliency heatmap comparison
  Rows = 3 models (JEPA+Jitter, VICReg-only, MAE)
  Cols = 4 representative sequences (one per complexity class)
  Color = gradient magnitude (GradCAM-style)

Fig B: Per-command-type average saliency (bar chart)
  Shows which command types each model attends to
  Key finding: JEPA focuses on SOL/EXT (structural),
               MAE focuses on curve args (reconstruction)

Fig C: Block influence heatmap (LOO analysis)
  For multi-block sequences, cosine distance when each block removed
  Confirms early-middle blocks dominate shape identity

Saves to {PROJ}/figures/
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

sys.path.insert(0, '/content/DeepCAD_prashant')
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader
from model.model_utils import (_get_key_padding_mask,
                                _get_group_mask,
                                _get_padding_mask)
from cadlib.macro import (EXT_IDX, SOL_IDX, LINE_IDX,
                           ARC_IDX, CIRCLE_IDX, EOS_IDX)

PROJ      = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT = '/content/deepcad_data'
FIGS_DIR  = f'{PROJ}/figures'
LABELS    = '/content/test_labels.npy'
os.makedirs(FIGS_DIR, exist_ok=True)

CMD_NAMES = {SOL_IDX: 'SOL', LINE_IDX: 'LINE', ARC_IDX: 'ARC',
             CIRCLE_IDX: 'CIR', EXT_IDX: 'EXT', EOS_IDX: 'EOS'}
CMD_COLORS = {SOL_IDX: '#E8651A', LINE_IDX: '#4A90D9', ARC_IDX: '#7BC67E',
              CIRCLE_IDX: '#9B59B6', EXT_IDX: '#E74C3C', EOS_IDX: '#95A5A6'}

MODELS = {
    'H-CAD-JEPA+Jitter': {
        'ckpt':          f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt',
        'use_group_emb': True,
        'key':           'ema_encoder',
        'color':         '#2ECC71',
    },
    'VICReg-only': {
        'ckpt':          f'{PROJ}/hcadjepa_vicreg_only/model/latest.pt',
        'use_group_emb': True,
        'key':           'encoder',
        'color':         '#E74C3C',
    },
    'MAE-on-CAD': {
        'ckpt':          f'{PROJ}/hcadjepa_mae/model/latest.pt',
        'use_group_emb': True,
        'key':           'ema_encoder',
        'color':         '#3498DB',
    },
}


# ── Encoder loading ───────────────────────────────────────────

def make_cfg(use_group_emb=True):
    return argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.0, use_group_emb=use_group_emb,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=1, num_workers=0, data_root=DATA_ROOT, use_cls=False,
    )


def load_encoder(ckpt_path, use_group_emb, key):
    cfg     = make_cfg(use_group_emb)
    encoder = JEPAEncoder(cfg).cuda().float()
    raw     = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    encoder.load_state_dict(raw[key])
    encoder.eval()
    return encoder, cfg


# ── GradCAM-style token saliency ─────────────────────────────

# def compute_gradient_saliency(encoder, commands, args):
#     """
#     GradCAM-style: saliency_i = ||∂||z|| / ∂h_i||
#     where h_i is the i-th token's initial embedding vector.

#     commands: (1, S) long
#     args:     (1, S, 16) long
#     Returns:  (S,) numpy saliency per token
#     """
#     encoder.eval()
#     commands_sf = commands.permute(1, 0)   # (S, 1)
#     args_sf     = args.permute(1, 0, 2)   # (S, 1, 16)

#     kpm       = _get_key_padding_mask(commands_sf, seq_dim=0)
#     grp_mask  = (_get_group_mask(commands_sf, seq_dim=0)
#                  if encoder.use_group else None)

#     # Get initial embedding — this is our "feature map" analog
#     src = encoder.embedding(commands_sf, args_sf, grp_mask).float()
#     src.requires_grad_(True)

#     # Forward through transformer (no masking for saliency)
#     out     = encoder.encoder(src, mask=None, src_key_padding_mask=kpm)
#     out     = encoder.output_norm(out)

#     # Mean pool over non-EOS positions
#     pad_mask = _get_padding_mask(commands_sf, seq_dim=0)  # (S, 1, 1)
#     z = (out * pad_mask).sum(0) / pad_mask.sum(0).clamp(min=1)  # (1, d)

#     # Scalar = L2 norm of embedding
#     score = z.norm(dim=-1).sum()
#     score.backward()

#     # Saliency = gradient magnitude at each token
#     saliency = src.grad.norm(dim=-1).squeeze(1)  # (S,)
#     saliency = saliency / (saliency.max() + 1e-8)
#     return saliency.detach().cpu().numpy()

def compute_gradient_saliency(encoder, commands, args):
    """
    GradCAM-style: saliency_i = ||∂||z|| / ∂h_i||
    where h_i is the i-th token's initial embedding vector.
    """
    encoder.eval()
    commands_sf = commands.permute(1, 0)
    args_sf     = args.permute(1, 0, 2)

    kpm      = _get_key_padding_mask(commands_sf, seq_dim=0)
    grp_mask = (_get_group_mask(commands_sf, seq_dim=0)
                if encoder.use_group else None)

    # Step 1: compute embedding with no_grad (just getting values)
    with torch.no_grad():
        src_base = encoder.embedding(commands_sf, args_sf, grp_mask).float()

    # Step 2: create a LEAF tensor — gradient will accumulate here
    src = src_base.clone().detach().requires_grad_(True)

    # Step 3: forward through transformer (no no_grad — need grad flow)
    out      = encoder.encoder(src.half() if next(encoder.parameters()).dtype == torch.float16
                               else src,
                               mask=None, src_key_padding_mask=kpm)
    out      = encoder.output_norm(out).float()

    # Step 4: mean pool over non-EOS positions
    pad_mask = _get_padding_mask(commands_sf, seq_dim=0)  # (S, 1, 1)
    z        = (out * pad_mask).sum(0) / pad_mask.sum(0).clamp(min=1)  # (1, d)

    # Step 5: scalar → backward
    score = z.norm(dim=-1).sum()
    score.backward()

    # Step 6: saliency = gradient magnitude per token
    assert src.grad is not None, "Gradient is None — check encoder dtype"
    saliency = src.grad.float().norm(dim=-1).squeeze(1)  # (S,)
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency.detach().cpu().numpy()


# ── Sequence selection ────────────────────────────────────────

def select_representative_sequences(n_per_class=1):
    """Pick one representative sequence per complexity class."""
    with open(f'{DATA_ROOT}/train_val_test_split.json') as f:
        test_ids = json.load(f)['test']
    labels = np.load(LABELS)

    selected = []
    for cls in range(4):
        cls_idx = np.where(labels == cls)[0]
        # Pick sequences with moderate length (not too short, not too long)
        for idx in cls_idx[::len(cls_idx)//20]:  # sample spread
            h5p = f"{DATA_ROOT}/cad_vec/{test_ids[idx]}.h5"
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as f:
                    vec = f['vec'][:]
                n_ext = (vec[:, 0] == EXT_IDX).sum()
                seq_len = (vec[:, 0] != EOS_IDX).sum()
                # Pick sequences with reasonable complexity
                if 5 <= seq_len <= 40:
                    selected.append({
                        'seq_id': test_ids[idx],
                        'label':  int(labels[idx]),
                        'n_ext':  int(n_ext),
                        'len':    int(seq_len),
                    })
                    break
    return selected


def load_sequence(seq_id, cfg):
    """Load h5 sequence as tensors."""
    h5p = f"{DATA_ROOT}/cad_vec/{seq_id}.h5"
    with h5py.File(h5p, 'r') as f:
        vec = f['vec'][:]

    # Pad to max_total_len
    from cadlib.macro import EOS_VEC
    pad_len = cfg.max_total_len - vec.shape[0]
    if pad_len > 0:
        pad = np.tile(EOS_VEC[np.newaxis], (pad_len, 1))
        vec = np.concatenate([vec, pad], axis=0)

    cmd  = torch.tensor(vec[:, 0], dtype=torch.long).unsqueeze(0).cuda()
    args = torch.tensor(vec[:, 1:], dtype=torch.long).unsqueeze(0).cuda()
    return cmd, args, vec[:, 0]


# ══════════════════════════════════════════════════════════════
#  Figure A: Saliency Heatmap Comparison
# ══════════════════════════════════════════════════════════════

def make_saliency_heatmap(sequences, model_results):
    """
    Rows = models, Cols = sequences
    Each cell: colored bars per token (color=command type, alpha=saliency)
    """
    n_models = len(model_results)
    n_seqs   = len(sequences)

    fig, axes = plt.subplots(n_models, n_seqs,
                              figsize=(n_seqs * 4.5, n_models * 2.2),
                              facecolor='white')

    class_names = ['1-op', '2-ops', '3-5-ops', '6+-ops']

    for row, (mname, sal_data) in enumerate(model_results.items()):
        for col, seq in enumerate(sequences):
            ax   = axes[row][col] if n_models > 1 else axes[col]
            cmd_seq   = seq['cmd_np']
            saliency  = sal_data[col]
            seq_len   = (cmd_seq != EOS_IDX).sum()

            # Plot one bar per token
            xs = np.arange(seq_len)
            cs = [CMD_COLORS.get(int(c), '#AAAAAA') for c in cmd_seq[:seq_len]]
            sal_vals = saliency[:seq_len]

            bars = ax.bar(xs, sal_vals, color=cs,
                          edgecolor='none', width=0.85)

            # Mark block boundaries (SOL tokens)
            for i, c in enumerate(cmd_seq[:seq_len]):
                if int(c) == SOL_IDX:
                    ax.axvline(i - 0.4, color='gray', lw=0.5,
                               linestyle='--', alpha=0.5)

            ax.set_xlim(-0.5, seq_len - 0.5)
            ax.set_ylim(0, 1.15)
            ax.set_yticks([])
            ax.set_xticks([])

            # Column header (sequence info)
            if row == 0:
                ax.set_title(f'Class {seq["label"]} ({class_names[seq["label"]]})\n'
                             f'{seq["n_ext"]} ops, len={seq["len"]}',
                             fontsize=9, fontweight='bold', pad=4)

            # Row label (model name)
            if col == 0:
                ax.set_ylabel(mname, fontsize=9, fontweight='bold',
                              rotation=0, ha='right', va='center',
                              labelpad=100)

            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')

    # Legend for command types
    patches = [mpatches.Patch(color=CMD_COLORS[c], label=CMD_NAMES[c])
               for c in [SOL_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX]]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, frameon=False,
               title='Token type  (bar height = saliency)',
               title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('GradCAM-style Token Saliency: What does each model attend to?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    out = f'{FIGS_DIR}/fig_saliency_heatmap.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure B: Per-Command-Type Average Saliency
# ══════════════════════════════════════════════════════════════

def compute_commandtype_saliency(encoder, n_seqs=200):
    """
    Average saliency per command type over many sequences.
    Returns dict: {cmd_idx: mean_saliency}
    """
    with open(f'{DATA_ROOT}/train_val_test_split.json') as f:
        test_ids = json.load(f)['test']
    labels = np.load(LABELS)
    cfg = make_cfg(encoder.use_group)

    cmd_saliency_sum   = {c: 0.0 for c in CMD_NAMES}
    cmd_saliency_count = {c: 0   for c in CMD_NAMES}

    np.random.seed(42)
    idx_sample = np.random.choice(len(test_ids), n_seqs, replace=False)

    for idx in tqdm(idx_sample, desc='  Averaging saliency', leave=False):
        try:
            cmd, args, cmd_np = load_sequence(test_ids[idx], cfg)
            sal = compute_gradient_saliency(encoder, cmd, args)
            seq_len = (cmd_np != EOS_IDX).sum()
            for i in range(seq_len):
                c = int(cmd_np[i])
                if c in cmd_saliency_sum:
                    cmd_saliency_sum[c]   += float(sal[i])
                    cmd_saliency_count[c] += 1
        except Exception:
            continue

    result = {}
    for c in CMD_NAMES:
        if cmd_saliency_count[c] > 0:
            result[c] = cmd_saliency_sum[c] / cmd_saliency_count[c]
        else:
            result[c] = 0.0
    return result


def make_command_saliency_chart(model_cmd_saliency):
    """
    Grouped bar chart: x=command types, groups=models, y=mean saliency
    """
    cmds    = [SOL_IDX, EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX]
    names   = [CMD_NAMES[c] for c in cmds]
    n_mod   = len(model_cmd_saliency)
    x       = np.arange(len(cmds))
    width   = 0.22
    offsets = np.linspace(-(n_mod-1)*width/2, (n_mod-1)*width/2, n_mod)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')

    for i, (mname, sal) in enumerate(model_cmd_saliency.items()):
        vals = [sal.get(c, 0) for c in cmds]
        # Normalize to sum=1 for fair comparison
        tot = sum(vals) + 1e-8
        vals = [v / tot for v in vals]
        color = MODELS[mname]['color']
        ax.bar(x + offsets[i], vals, width, label=mname,
               color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Relative saliency (normalized)', fontsize=10)
    ax.set_title('Per-Token-Type Average Saliency\n'
                 'JEPA focuses on structural tokens (SOL/EXT); '
                 'MAE focuses on geometry tokens',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate the key insight
    ax.annotate('Structural\nboundaries', xy=(0, 0), xytext=(0, 0),
                fontsize=7.5, color='#666666', ha='center')

    plt.tight_layout()
    out = f'{FIGS_DIR}/fig_saliency_by_command.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Figure C: Block LOO Influence Heatmap
# ══════════════════════════════════════════════════════════════

def compute_block_loo_influence(encoder, n_seqs=100, min_blocks=3):
    """
    For multi-block sequences: remove each block, measure embedding displacement.
    Returns: matrix of shape (max_blocks, n_seqs) with displacement values.
    """
    from cadlib.macro import EOS_VEC
    with open(f'{DATA_ROOT}/train_val_test_split.json') as f:
        test_ids = json.load(f)['test']
    labels = np.load(LABELS)
    cfg    = make_cfg(encoder.use_group)

    # Pick multi-block test sequences
    cls3_idx = np.where(labels == 3)[0]
    np.random.seed(42)
    selected = cls3_idx[np.random.choice(len(cls3_idx), min(n_seqs*3, len(cls3_idx)), replace=False)]

    results = []  # list of (n_blocks, [loo_displacements])

    for idx in selected:
        try:
            h5p = f"{DATA_ROOT}/cad_vec/{test_ids[idx]}.h5"
            with h5py.File(h5p, 'r') as f:
                vec = f['vec'][:]

            cmds_np = vec[:, 0]
            # Find blocks (SOL...EXT)
            blocks, sol = [], None
            for i, c in enumerate(cmds_np):
                if c == SOL_IDX:   sol = i
                elif c == EXT_IDX and sol is not None:
                    blocks.append((sol, i)); sol = None
                elif c == EOS_IDX: break

            if len(blocks) < min_blocks: continue

            # Baseline embedding
            cmd, args, _ = load_sequence(test_ids[idx], cfg)
            with torch.no_grad():
                mem, _ = encoder(cmd, args)
                pad    = _get_padding_mask(cmd.permute(1,0), seq_dim=0)
                z_base = (mem * pad).sum(0) / pad.sum(0).clamp(min=1)

            displacements = []
            for (s, e) in blocks:
                # Mask this block out
                cmd_abl = cmd.clone()
                cmd_abl[0, s:e+1] = EOS_IDX
                args_abl = args.clone()
                args_abl[0, s:e+1] = -1
                with torch.no_grad():
                    mem_abl, _ = encoder(cmd_abl, args_abl)
                    pad_abl    = _get_padding_mask(cmd_abl.permute(1,0), seq_dim=0)
                    z_abl      = (mem_abl * pad_abl).sum(0) / pad_abl.sum(0).clamp(min=1)
                dist = (1 - F.cosine_similarity(z_base, z_abl)).item()
                displacements.append(dist)

            results.append(displacements)
            if len(results) >= n_seqs: break

        except Exception:
            continue

    return results


def make_block_influence_heatmap(loo_results, model_name):
    """
    Left: block influence by absolute position (heatmap, max 8 blocks)
    Right: block influence by relative position (line plot)
    """
    max_blocks = 8
    # Pad to max_blocks
    matrix = np.zeros((len(loo_results), max_blocks))
    for i, disp in enumerate(loo_results):
        n = min(len(disp), max_blocks)
        matrix[i, :n] = disp[:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')

    # Left: sorted heatmap
    # Sort by n_blocks
    lengths = [(matrix[i] > 0).sum() for i in range(len(matrix))]
    order   = np.argsort(lengths)
    mat_sorted = matrix[order]

    im = ax1.imshow(mat_sorted, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=matrix.max())
    ax1.set_xlabel('Block index', fontsize=10)
    ax1.set_ylabel(f'Sequence (n={len(loo_results)})', fontsize=10)
    ax1.set_title('Block LOO Influence Heatmap\n(yellow=high impact)', fontsize=10)
    ax1.set_xticks(range(max_blocks))
    ax1.set_xticklabels([f'B{i+1}' for i in range(max_blocks)])
    plt.colorbar(im, ax=ax1, fraction=0.046, label='Cosine displacement')

    # Right: mean influence by relative position
    rel_bins = 5
    rel_means = np.zeros(rel_bins)
    rel_stds  = np.zeros(rel_bins)
    rel_vals  = [[] for _ in range(rel_bins)]

    for disp in loo_results:
        n = len(disp)
        for i, d in enumerate(disp):
            bin_idx = min(int(i / n * rel_bins), rel_bins - 1)
            rel_vals[bin_idx].append(d)

    for b in range(rel_bins):
        if rel_vals[b]:
            rel_means[b] = np.mean(rel_vals[b])
            rel_stds[b]  = np.std(rel_vals[b])

    xs = np.linspace(0, 1, rel_bins)
    ax2.plot(xs, rel_means, 'o-', color='#E74C3C', linewidth=2, markersize=7)
    ax2.fill_between(xs, rel_means - rel_stds, rel_means + rel_stds,
                     alpha=0.2, color='#E74C3C')
    ax2.set_xlabel('Relative block position (0=first, 1=last)', fontsize=10)
    ax2.set_ylabel('Mean cosine displacement', fontsize=10)
    ax2.set_title('Block Influence by Relative Position\n'
                  'Early-middle blocks dominate shape identity',
                  fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['Start', '', 'Mid', '', 'End'])

    fig.suptitle(f'Block Leave-One-Out Analysis — {model_name}',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{FIGS_DIR}/fig_block_loo.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Explainability Figures")
    print("=" * 60)

    # Select representative sequences
    print("\nSelecting representative sequences...")
    sequences = select_representative_sequences(n_per_class=1)

    print(f"  Selected {len(sequences)} sequences:")
    print([f"cls{s['label']}({s['n_ext']}ops)" for s in sequences])
    # Load models
    encoders = {}
    for mname, mcfg in MODELS.items():
        print(f"\nLoading {mname}...")
        enc, cfg = load_encoder(mcfg['ckpt'], mcfg['use_group_emb'], mcfg['key'])
        encoders[mname] = (enc, cfg)
        print(f"  ✓ Loaded")

    # Load sequences as tensors
    print("\nLoading sequences...")
    cfg_default = make_cfg()
    for seq in sequences:
        cmd, args, cmd_np = load_sequence(seq['seq_id'], cfg_default)
        seq['cmd']    = cmd
        seq['args']   = args
        seq['cmd_np'] = cmd_np

    # ── Figure A: Saliency heatmap ────────────────────────────
    print("\nComputing gradient saliency (Fig A)...")
    model_saliency = {}
    for mname, (enc, cfg) in encoders.items():
        sals = []
        for seq in sequences:
            sal = compute_gradient_saliency(enc, seq['cmd'], seq['args'])
            sals.append(sal)
        model_saliency[mname] = sals
        print(f"  ✓ {mname}")

    make_saliency_heatmap(sequences, model_saliency)

    # ── Figure B: Per-command-type saliency ───────────────────
    print("\nComputing per-command saliency (Fig B)...")
    model_cmd_sal = {}
    for mname, (enc, cfg) in encoders.items():
        model_cmd_sal[mname] = compute_commandtype_saliency(enc, n_seqs=150)
        print(f"  ✓ {mname}")

    make_command_saliency_chart(model_cmd_sal)

    # ── Figure C: Block LOO ───────────────────────────────────
    print("\nComputing block LOO influence (Fig C)...")
    best_enc = encoders['H-CAD-JEPA+Jitter'][0]
    loo = compute_block_loo_influence(best_enc, n_seqs=80)
    print(f"  Computed LOO for {len(loo)} sequences")
    make_block_influence_heatmap(loo, 'H-CAD-JEPA+Jitter')

    print(f"\n✓ All explainability figures saved to {FIGS_DIR}")


if __name__ == '__main__':
    main()