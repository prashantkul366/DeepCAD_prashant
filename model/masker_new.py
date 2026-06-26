import random
import torch
import numpy as np
from cadlib.macro import (
    EOS_IDX, SOL_IDX, EXT_IDX,
    LINE_IDX, ARC_IDX, CIRCLE_IDX
)

SKETCH_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}


# ── Block finder ──────────────────────────────────────────────────────────────

def find_blocks(cmd_np):
    """
    cmd_np: 1D numpy array of command indices.
    Returns list of (start, end) inclusive per SOL→EXT block.
    """
    blocks, sol = [], None
    for i, c in enumerate(cmd_np.tolist()):
        if c == SOL_IDX:
            sol = i
        elif c == EXT_IDX and sol is not None:
            blocks.append((sol, i))
            sol = None
        elif c == EOS_IDX:
            break
    return blocks


# ── Token masking ─────────────────────────────────────────────────────────────

def _apply_token_masking(cmd_np, mask_row, mask_ratio=0.50):
    """
    Mask curve tokens (LINE/ARC/CIRCLE) only.
    SOL, EXT, EOS are never masked — they are structural anchors.

    Enforces:
      - min 2 tokens masked (real learning signal)
      - min 1 token visible (context always exists)

    For sequences with only 1 curve: masks that 1 curve.
    The SOL+EXT pair still provides structural context.
    """
    curve_pos = [i for i, c in enumerate(cmd_np.tolist())
                 if int(c) in SKETCH_CMDS]
    n_curves = len(curve_pos)
    if n_curves == 0:
        return

    # Target number to mask
    n_mask = int(round(n_curves * mask_ratio))

    # Enforce bounds: min 1, max n_curves-1 (keep at least 1 visible)
    # For n_curves==1: mask it — SOL+EXT is still sufficient context
    n_mask = max(1, min(n_mask, n_curves))
    if n_curves > 1:
        n_mask = min(n_mask, n_curves - 1)  # keep at least 1 visible

    for pos in random.sample(curve_pos, n_mask):
        mask_row[pos] = True


# ── Block masking ─────────────────────────────────────────────────────────────

def _apply_anchor_first_block_masking(blocks, mask_row, mask_ratio=0.40):
    """
    Block 0 is always kept as the geometric anchor.
    Masks from blocks[1:] only.

    For 2-block sequences: masks block 1, block 0 is context.
    For 3+ block sequences: masks ceil(n_remaining * mask_ratio),
      keeping at least 1 non-anchor block visible when n_remaining >= 2.

    Called only when len(blocks) >= 2.
    """
    n_blocks = len(blocks)
    candidates = list(range(1, n_blocks))   # never includes block 0
    n_candidates = len(candidates)

    # How many to mask from candidates
    n_mask = max(1, int(np.ceil(n_candidates * mask_ratio)))

    # Keep at least 1 non-anchor visible when possible
    if n_candidates > 1:
        n_mask = min(n_mask, n_candidates - 1)
    else:
        # Only 1 candidate (2-block sequence): mask it, block 0 is context
        n_mask = 1

    for idx in random.sample(candidates, n_mask):
        s, e = blocks[idx]
        mask_row[s:e + 1] = True


# ── Masker classes ────────────────────────────────────────────────────────────

class TokenLevelMasker:
    """
    Pure token masking. Used for MAE baseline.
    Works on all sequences including single-block.
    mask_ratio applied to curve tokens only.
    """
    def __init__(self, mask_ratio=0.50):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        """
        commands: (N, S) long tensor
        Returns: (N, S) bool tensor, True = masked
        """
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        for i in range(N):
            _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio)
        return torch.from_numpy(mask_np).to(commands.device)


class AnchorFirstMasker:
    """
    JEPA masker. Block 0 is always the geometric anchor — never masked.

    Dispatch:
      Single-block (n_blocks == 1): token masking fallback at mask_ratio_token.
        No alternative — block masking impossible without losing all context.
      Multi-block  (n_blocks >= 2): anchor-first block masking.
        Block 0 kept, mask from blocks[1:] at mask_ratio.

    Justified by EDA:
      - Position 0 mean 4.05 curves vs 3.14 at position 1 — structurally distinct
      - anchor-first masking justified: YES confirmed by full dataset analysis
    """
    def __init__(self, mask_ratio=0.40, mask_ratio_token=0.50):
        self.mask_ratio       = mask_ratio
        self.mask_ratio_token = mask_ratio_token

    def __call__(self, commands):
        """
        commands: (N, S) long tensor
        Returns:
            target_mask:   (N, S) bool — True = masked position
            regime_per_seq: list of str — 'token' or 'block' per sequence
        """
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        regime_per_seq = []

        for i in range(N):
            blocks = find_blocks(cmd_np[i])

            if len(blocks) >= 2:
                _apply_anchor_first_block_masking(
                    blocks, mask_np[i], self.mask_ratio
                )
                regime_per_seq.append('block')
            else:
                _apply_token_masking(
                    cmd_np[i], mask_np[i], self.mask_ratio_token
                )
                regime_per_seq.append('token')

        target_mask = torch.from_numpy(mask_np).to(commands.device)
        return target_mask, regime_per_seq


# ── Factory ───────────────────────────────────────────────────────────────────

def get_masker(strategy, cfg):
    """
    strategy: 'anchor_block' (JEPA) | 'token' (MAE)
    """
    tok_ratio = getattr(cfg, 'mask_ratio_token', 0.50)
    blk_ratio = getattr(cfg, 'mask_ratio',       0.40)

    if strategy == 'token':
        return TokenLevelMasker(mask_ratio=tok_ratio)
    elif strategy == 'anchor_block':
        return AnchorFirstMasker(
            mask_ratio=blk_ratio,
            mask_ratio_token=tok_ratio
        )
    else:
        raise ValueError(
            f"Unknown masking strategy: '{strategy}'. "
            f"Use 'anchor_block' (JEPA) or 'token' (MAE)."
        )