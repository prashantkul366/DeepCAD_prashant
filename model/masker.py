import random
import torch
import numpy as np
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX

SKETCH_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}


# ─────────────────────────────────────────────────────────────
# Core utility
# ─────────────────────────────────────────────────────────────

def find_blocks(cmd_seq):
    """
    cmd_seq: 1D numpy array of command indices (length S)
    Returns list of (start, end) — inclusive, one per SOL...Ext block.
    """
    blocks, sol = [], None
    for i, c in enumerate(cmd_seq.tolist()):
        if c == SOL_IDX:
            sol = i
        elif c == EXT_IDX and sol is not None:
            blocks.append((sol, i))
            sol = None
        elif c == EOS_IDX:
            break
    return blocks


# ─────────────────────────────────────────────────────────────
# Level 1 — Token masking (works on ALL shapes incl. 1-block)
# ─────────────────────────────────────────────────────────────

def _apply_token_masking(cmd_np_i, mask_i, mask_ratio=0.40):
    """
    Mask individual curve tokens (Line/Arc/Circle) within sketch regions.
    SOL and Ext tokens are structural anchors — never masked.
    Works correctly even for single-block shapes.
    """
    curve_pos = [j for j, c in enumerate(cmd_np_i) if int(c) in SKETCH_CMDS]
    if not curve_pos:
        return  # no curves to mask (shouldn't happen in real data)
    n_mask = max(1, int(round(len(curve_pos) * mask_ratio)))
    for pos in random.sample(curve_pos, min(n_mask, len(curve_pos))):
        mask_i[pos] = True


# ─────────────────────────────────────────────────────────────
# Level 2 — Block masking (requires >= 2 blocks)
# ─────────────────────────────────────────────────────────────

def _apply_block_masking(blocks, mask_i, mask_ratio=0.40):
    """
    Mask complete SOL...Ext operation blocks.
    Always keeps at least one block as context.
    Caller guarantees len(blocks) >= 2.
    """
    n_blocks = len(blocks)
    n_mask = max(1, min(int(np.ceil(n_blocks * mask_ratio)), n_blocks - 1))
    for idx in random.sample(range(n_blocks), n_mask):
        s, e = blocks[idx]
        mask_i[s:e + 1] = True


# ─────────────────────────────────────────────────────────────
# Level 3 — Group masking (requires >= 3 blocks)
# ─────────────────────────────────────────────────────────────

def _apply_group_masking(blocks, mask_i, n_groups=2):
    """
    Mask n_groups consecutive operation blocks.
    Caller guarantees len(blocks) >= n_groups + 1.
    """
    n_blocks = len(blocks)
    start_b = random.randint(0, n_blocks - n_groups)
    for b in range(start_b, start_b + n_groups):
        s, e = blocks[b]
        mask_i[s:e + 1] = True


# ─────────────────────────────────────────────────────────────
# Adaptive masker — per-sequence level selection
# ─────────────────────────────────────────────────────────────

class AdaptiveMasker:
    """
    Selects masking level per sequence based on block count.
    This ensures every shape contributes gradient signal:
      n_blocks == 1 → token only
      n_blocks == 2 → token or block (50/50)
      n_blocks >= 3 → token, block, or group (33/33/33)

    Returns (target_mask, level_per_seq) where:
      target_mask:   (N, S) bool tensor
      level_per_seq: list of str, one per sequence in batch
    """
    def __init__(self, mask_ratio=0.40, n_groups=2):
        self.mask_ratio = mask_ratio
        self.n_groups   = n_groups

    def __call__(self, commands):
        """commands: (N, S) long tensor"""
        N, S = commands.shape
        target_mask    = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        level_per_seq  = []
        cmd_np         = commands.cpu().numpy()
        mask_np        = target_mask.cpu().numpy()

        for i in range(N):
            blocks  = find_blocks(cmd_np[i])
            n_blocks = len(blocks)

            # ── Select level based on block count ──────────
            if n_blocks == 1:
                level = 'token'
            elif n_blocks == 2:
                level = random.choice(['token', 'block'])
            else:
                level = random.choices(
                    ['token', 'block', 'group'],
                    weights=[0.33, 0.34, 0.33]
                )[0]

            # ── Apply masking ───────────────────────────────
            if level == 'token':
                _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio)
            elif level == 'block':
                _apply_block_masking(blocks, mask_np[i], self.mask_ratio)
            elif level == 'group':
                if n_blocks >= self.n_groups + 1:
                    _apply_group_masking(blocks, mask_np[i], self.n_groups)
                else:
                    # Fallback — shouldn't reach here given level selection above
                    _apply_block_masking(blocks, mask_np[i], self.mask_ratio)

            level_per_seq.append(level)

        # Write back to GPU tensor
        target_mask = torch.from_numpy(mask_np).to(commands.device)
        return target_mask, level_per_seq


# ─────────────────────────────────────────────────────────────
# Fixed single-level maskers (for T2 ablation runs)
# Each handles single-block shapes gracefully via token fallback
# ─────────────────────────────────────────────────────────────

class TokenLevelMasker:
    """Pure token masking. Works on all shapes."""
    def __init__(self, mask_ratio=0.40):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np  = commands.cpu().numpy()
        mask_np = target_mask.cpu().numpy()
        for i in range(N):
            _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio)
        return torch.from_numpy(mask_np).to(commands.device)


class OperationBlockMasker:
    """
    Block masking with token fallback for single-block shapes.
    Used for CAD-JEPA (block-only) and T2 ablation.
    """
    def __init__(self, mask_ratio=0.40):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np  = commands.cpu().numpy()
        mask_np = target_mask.cpu().numpy()
        for i in range(N):
            blocks   = find_blocks(cmd_np[i])
            n_blocks = len(blocks)
            if n_blocks >= 2:
                _apply_block_masking(blocks, mask_np[i], self.mask_ratio)
            else:
                # Single-block fallback: token masking
                _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio)
        return torch.from_numpy(mask_np).to(commands.device)


class GroupLevelMasker:
    """
    Group masking with graceful fallbacks.
    n_blocks >= 3: group masking
    n_blocks == 2: block masking
    n_blocks == 1: token masking
    """
    def __init__(self, n_groups=2):
        self.n_groups = n_groups

    def __call__(self, commands):
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np  = commands.cpu().numpy()
        mask_np = target_mask.cpu().numpy()
        for i in range(N):
            blocks   = find_blocks(cmd_np[i])
            n_blocks = len(blocks)
            if n_blocks >= self.n_groups + 1:
                _apply_group_masking(blocks, mask_np[i], self.n_groups)
            elif n_blocks >= 2:
                _apply_block_masking(blocks, mask_np[i], mask_ratio=0.40)
            else:
                _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.40)
        return torch.from_numpy(mask_np).to(commands.device)


# ─────────────────────────────────────────────────────────────
# Factory — used by trainerJEPA
# ─────────────────────────────────────────────────────────────

def get_masker(strategy, cfg):
    if strategy == 'block':
        return OperationBlockMasker(mask_ratio=cfg.mask_ratio)
    elif strategy == 'token':
        return TokenLevelMasker(mask_ratio=cfg.mask_ratio)
    elif strategy == 'group':
        return GroupLevelMasker(n_groups=cfg.n_mask_groups)
    elif strategy == 'hierarchical':
        return AdaptiveMasker(mask_ratio=cfg.mask_ratio, n_groups=cfg.n_mask_groups)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")