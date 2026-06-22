"""
figure_architecture.py  — Run anywhere (no GPU needed)
=========================================================
Creates the H-CAD-JEPA architecture diagram.

Layout:
  Left:   CAD sequence input + Adaptive Masker decision logic
  Center: Context Encoder || EMA Target Encoder (with arrows)
  Right:  Hierarchical Predictor (3 heads) + VICReg + Loss

Color scheme:
  Orange: inputs
  Blue:   context encoder path
  Teal:   target encoder path (EMA)
  Purple: predictor heads
  Red:    loss

Saves to {PROJ}/figures/fig_architecture.pdf + .png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

PROJ     = '/content/drive/MyDrive/jepa_experiments'
FIGS_DIR = f'{PROJ}/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────
C_INPUT    = '#F5A623'   # orange
C_MASK     = '#D4AC0D'   # yellow-orange
C_CONTEXT  = '#2E86C1'   # blue
C_TARGET   = '#1A9D7B'   # teal
C_PRED     = '#7D3C98'   # purple
C_VICREG   = '#C0392B'   # red
C_LOSS     = '#922B21'   # dark red
C_EMA      = '#17A589'   # EMA arrow
C_ARROW    = '#555555'   # regular arrows
C_BG       = '#F8F9FA'   # light background


def box(ax, x, y, w, h, text, color, fontsize=9, text_color='white',
        style='round,pad=0.1', alpha=1.0, bold=False):
    """Draw a rounded box with centered text."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=style,
                           facecolor=color, edgecolor='white',
                           linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_color,
            fontweight=weight, zorder=4, wrap=True,
            multialignment='center')


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, label='',
          style='->', lw=1.5, shrink=3):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label, fontsize=7.5, color=color,
                ha='left', va='center', style='italic')


def curved_arrow(ax, x1, y1, x2, y2, color=C_ARROW, rad=0.3, label='', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + rad * 0.3
        ax.text(mx, my, label, fontsize=8, color=color,
                ha='center', va='center', style='italic',
                fontweight='bold')


def make_architecture_diagram():
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor('white')

    # ── Title ─────────────────────────────────────────────────
    ax.text(0.5, 0.97, 'H-CAD-JEPA: Hierarchical Masked Prediction for CAD Retrieval',
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    # ══════════════════════════════════════════════════════════
    #  LEFT PANEL: Input + Adaptive Masker
    # ══════════════════════════════════════════════════════════

    # Input sequence visualization
    seq_tokens = ['SOL', 'LINE', 'ARC', 'EXT', 'SOL', 'CIR', 'EXT', 'EOS']
    tok_colors = {'SOL': '#E8651A', 'LINE': '#4A90D9', 'ARC': '#7BC67E',
                  'CIR': '#9B59B6', 'EXT': '#E74C3C', 'EOS': '#95A5A6'}
    tok_w = 0.055
    tok_h = 0.045
    tok_y = 0.82
    tok_x_start = 0.04

    ax.text(tok_x_start, tok_y + 0.055, 'CAD Sequence Input',
            fontsize=9, fontweight='bold', color='#333333')

    for i, tok in enumerate(seq_tokens):
        tx = tok_x_start + i * (tok_w + 0.005)
        color = tok_colors.get(tok, '#AAAAAA')
        box(ax, tx + tok_w/2, tok_y, tok_w, tok_h, tok,
            color, fontsize=7.5, bold=True)

    # Sequence info annotation
    ax.text(tok_x_start + 0.01, tok_y - 0.04,
            'cmd ∈ {SOL,LINE,ARC,CIR,EXT,EOS}  +  16 quantized args',
            fontsize=8, color='#666666', style='italic')

    # Adaptive Masker box
    masker_x, masker_y = 0.175, 0.62
    masker_w, masker_h = 0.20, 0.14
    box(ax, masker_x, masker_y, masker_w, masker_h,
        'Adaptive\nMasker', C_MASK, fontsize=10, bold=True)

    # Arrow: input → masker
    arrow(ax, 0.175, tok_y - tok_h/2 - 0.005, masker_x, masker_y + masker_h/2,
          color=C_ARROW)

    # Masker decision tree (small inset)
    inset_x = 0.01
    inset_y = 0.36
    inset_lines = [
        ('K = 1 block:',    'Token masking (50% curves)', '#F5A623'),
        ('K = 2 blocks:',   'Token (50%) or Block (50%)', '#E67E22'),
        ('K ≥ 3 blocks:',   'Token (25%), Block (50%),',  '#D35400'),
        ('',                '  Group (25%)',               '#D35400'),
    ]
    ax.text(inset_x, inset_y + 0.075, 'Masking Level Selection:',
            fontsize=8.5, fontweight='bold', color='#555555')
    for j, (key, val, col) in enumerate(inset_lines):
        yy = inset_y + 0.045 - j * 0.025
        ax.text(inset_x + 0.005, yy, key, fontsize=8, color='#333333',
                fontweight='bold' if key else 'normal')
        ax.text(inset_x + 0.08, yy, val, fontsize=8, color=col)

    # Draw a simple decision tree
    tree_x, tree_y = 0.17, 0.45
    # K=1 branch
    ax.plot([masker_x - 0.05, masker_x - 0.09],
            [masker_y - masker_h/2, tree_y + 0.03],
            color='#555555', lw=1.2)
    ax.text(masker_x - 0.11, tree_y + 0.02, 'K=1', fontsize=7.5,
            color='#555555', ha='right')
    # K=2 branch
    ax.plot([masker_x, masker_x],
            [masker_y - masker_h/2, tree_y + 0.03],
            color='#555555', lw=1.2)
    ax.text(masker_x + 0.01, tree_y + 0.02, 'K=2', fontsize=7.5,
            color='#555555', ha='left')
    # K≥3 branch
    ax.plot([masker_x + 0.05, masker_x + 0.09],
            [masker_y - masker_h/2, tree_y + 0.03],
            color='#555555', lw=1.2)
    ax.text(masker_x + 0.11, tree_y + 0.02, 'K≥3', fontsize=7.5,
            color='#555555', ha='left')

    # Three output modes
    modes = [
        (masker_x - 0.09, tree_y, 'Token\nmask',  '#F5A623'),
        (masker_x,         tree_y, 'Block\nmask',  '#E67E22'),
        (masker_x + 0.09, tree_y, 'Group\nmask',  '#D35400'),
    ]
    for mx, my, mlabel, mcolor in modes:
        box(ax, mx, my - 0.02, 0.06, 0.04, mlabel,
            mcolor, fontsize=7, text_color='white')

    # ══════════════════════════════════════════════════════════
    #  CENTER: Context Encoder + EMA Target Encoder
    # ══════════════════════════════════════════════════════════

    # Context path
    ctx_x, ctx_y = 0.42, 0.70
    ctx_w, ctx_h = 0.14, 0.12

    # Masked sequence box
    box(ax, ctx_x, ctx_y + 0.12, ctx_w, 0.06,
        'Context + Mask Tokens', C_INPUT, fontsize=8, bold=False)

    # Context encoder
    box(ax, ctx_x, ctx_y, ctx_w, ctx_h,
        'Context\nEncoder fθ\n(4L, 256d)', C_CONTEXT,
        fontsize=9, bold=True)

    # Arrow: masker → context input
    arrow(ax, masker_x + masker_w/2, masker_y,
          ctx_x - 0.01, ctx_y + 0.15,
          color=C_MASK, label='masked\nsequence')

    # Arrow: masked seq → context encoder
    arrow(ax, ctx_x, ctx_y + 0.09, ctx_x, ctx_y + ctx_h/2,
          color=C_CONTEXT)

    # VICReg annotation on context encoder
    ax.text(ctx_x + ctx_w/2 + 0.02, ctx_y + 0.02,
            'VICReg\n(λv=1.0, λc=0.04)',
            fontsize=7.5, color=C_VICREG, style='italic',
            ha='left', va='center')

    # Target encoder path
    tgt_x, tgt_y = 0.62, 0.70
    tgt_w, tgt_h = 0.14, 0.12

    # Full sequence box
    box(ax, tgt_x, tgt_y + 0.12, tgt_w, 0.06,
        'Full Sequence', C_INPUT, fontsize=8)

    # EMA target encoder
    box(ax, tgt_x, tgt_y, tgt_w, tgt_h,
        'Target\nEncoder fξ\n(EMA, no grad)', C_TARGET,
        fontsize=9, bold=True)

    # Arrow: input → target encoder
    arrow(ax, masker_x + masker_w/2 + 0.02, masker_y + 0.02,
          tgt_x - 0.01, tgt_y + 0.15,
          color=C_TARGET, label='full\nsequence')
    arrow(ax, tgt_x, tgt_y + 0.09, tgt_x, tgt_y + tgt_h/2,
          color=C_TARGET)

    # EMA update arrow (curved, from context back to target)
    curved_arrow(ax, ctx_x + ctx_w/2, ctx_y + 0.02,
                 tgt_x - tgt_w/2, tgt_y + 0.02,
                 color=C_EMA, rad=-0.35,
                 label='EMA update\nτ: 0.990→0.996', lw=2)

    # Context encoder output
    ctx_out_y = ctx_y - ctx_h/2 - 0.04
    box(ax, ctx_x, ctx_out_y, ctx_w, 0.055,
        'h_ctx\n(zeroed at targets)', C_CONTEXT,
        fontsize=8, alpha=0.85)
    arrow(ax, ctx_x, ctx_y - ctx_h/2,
          ctx_x, ctx_out_y + 0.03,
          color=C_CONTEXT)

    # Target encoder output
    tgt_out_y = tgt_y - tgt_h/2 - 0.04
    box(ax, tgt_x, tgt_out_y, tgt_w, 0.055,
        'z*\n(target embeddings)', C_TARGET,
        fontsize=8, alpha=0.85)
    arrow(ax, tgt_x, tgt_y - tgt_h/2,
          tgt_x, tgt_out_y + 0.03,
          color=C_TARGET)

    # ══════════════════════════════════════════════════════════
    #  RIGHT: Hierarchical Predictor + Loss
    # ══════════════════════════════════════════════════════════

    pred_cx = 0.82
    pred_heads = [
        (pred_cx, 0.75, 'Token Head\n(d_pred=128)', '#8E44AD'),
        (pred_cx, 0.62, 'Block Head\n(d_pred=128)', '#7D3C98'),
        (pred_cx, 0.49, 'Group Head\n(d_pred=128)', '#6C3483'),
    ]

    ax.text(pred_cx, 0.82, 'Hierarchical Predictor',
            ha='center', fontsize=10, fontweight='bold', color=C_PRED)
    ax.text(pred_cx, 0.785,
            '(3 independent cross-attention decoders)',
            ha='center', fontsize=8, color='#666666', style='italic')

    for px, py, plabel, pcolor in pred_heads:
        box(ax, px, py, 0.16, 0.08, plabel, pcolor,
            fontsize=8.5, bold=False)
        # Arrow from context output to each head
        arrow(ax, ctx_x + ctx_w/2, ctx_out_y,
              px - 0.08, py,
              color=C_CONTEXT, lw=1.2)

    # Predictor outputs → loss
    loss_y = 0.30
    box(ax, pred_cx, loss_y, 0.18, 0.08,
        'ℒ_pred\nSmooth-L1(ẑ, z*)',
        C_LOSS, fontsize=9, bold=True)

    for _, py, _, _ in pred_heads:
        arrow(ax, pred_cx, py - 0.04,
              pred_cx, loss_y + 0.04,
              color=C_LOSS, lw=1.2)

    # Arrows from target output to loss
    arrow(ax, tgt_x, tgt_out_y - 0.01,
          pred_cx + 0.01, loss_y + 0.04,
          color=C_TARGET, lw=1.2,
          label='z*\n(targets)')

    # VICReg loss
    vicreg_y = 0.30
    vicreg_x = 0.52
    box(ax, vicreg_x, vicreg_y, 0.13, 0.08,
        'ℒ_VICReg\n(variance +\ncovariance)',
        C_VICREG, fontsize=8.5, bold=True)

    arrow(ax, ctx_x, ctx_out_y - 0.01,
          vicreg_x, vicreg_y + 0.04,
          color=C_VICREG, lw=1.2, label='ctx tokens')

    # Total loss
    total_loss_x = (pred_cx + vicreg_x) / 2
    total_loss_y = 0.14
    box(ax, total_loss_x, total_loss_y, 0.26, 0.07,
        'ℒ = ℒ_pred + ℒ_VICReg',
        C_LOSS, fontsize=10, bold=True)

    arrow(ax, pred_cx, loss_y - 0.04,
          total_loss_x + 0.04, total_loss_y + 0.035,
          color=C_LOSS, lw=1.5)
    arrow(ax, vicreg_x, vicreg_y - 0.04,
          total_loss_x - 0.04, total_loss_y + 0.035,
          color=C_VICREG, lw=1.5)

    # ── Downstream: mean pooling ─────────────────────────────
    pool_x = 0.42
    pool_y = 0.14
    box(ax, pool_x, pool_y, 0.13, 0.06,
        'Mean Pool\n→ z ∈ ℝ²⁵⁶',
        '#1A252F', fontsize=8.5, bold=True)
    arrow(ax, ctx_x, ctx_out_y - 0.01,
          pool_x, pool_y + 0.03,
          color='#1A252F', lw=1.5,
          label='Downstream\nretrieval')

    # ── Legend ────────────────────────────────────────────────
    legend_items = [
        ('Input sequence',       C_INPUT),
        ('Context Encoder (θ)',  C_CONTEXT),
        ('Target Encoder (ξ)',   C_TARGET),
        ('Predictor heads',      C_PRED),
        ('VICReg',               C_VICREG),
        ('EMA update',           C_EMA),
    ]
    lx, ly = 0.02, 0.22
    ax.text(lx, ly + 0.025, 'Legend:', fontsize=9, fontweight='bold')
    for i, (label, color) in enumerate(legend_items):
        row, col = divmod(i, 3)
        lxi = lx + col * 0.13
        lyi = ly - row * 0.025
        ax.add_patch(FancyBboxPatch((lxi, lyi - 0.008), 0.015, 0.015,
                                     boxstyle='round,pad=0.02',
                                     facecolor=color, edgecolor='none'))
        ax.text(lxi + 0.02, lyi, label, fontsize=8, va='center', color='#333333')

    # ── Parameter counts ─────────────────────────────────────
    param_text = ('Encoder: 2.41M  |  Predictor: 2.61M  |  '
                  'Total: 5.02M params  |  500 epochs  |  A100')
    ax.text(0.5, 0.04, param_text,
            ha='center', fontsize=8.5, color='#666666',
            style='italic', transform=ax.transAxes)

    # ── Section dividers ─────────────────────────────────────
    for xv in [0.30, 0.50, 0.74]:
        ax.axvline(xv, color='#DDDDDD', lw=0.8, linestyle=':', zorder=1)

    section_labels = [
        (0.15, 0.93, 'Input &\nAdaptive Masker'),
        (0.40, 0.93, 'Context\nEncoder'),
        (0.62, 0.93, 'EMA Target\nEncoder'),
        (0.82, 0.93, 'Hierarchical\nPredictor'),
    ]
    for sx, sy, slabel in section_labels:
        ax.text(sx, sy, slabel, ha='center', fontsize=8.5,
                color='#888888', style='italic',
                transform=ax.transAxes)

    plt.tight_layout()

    # Save both formats
    for ext in ['png', 'pdf']:
        out = f'{FIGS_DIR}/fig_architecture.{ext}'
        plt.savefig(out, dpi=200 if ext == 'png' else 300,
                    bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved → {out}")

    plt.close()


if __name__ == '__main__':
    print("Creating architecture diagram...")
    make_architecture_diagram()
    print("Done.")