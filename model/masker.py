import random
import torch
import numpy as np
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX

SKETCH_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}


# ─────────────────────────────────────────────
#  Core utility: find SOL...Ext block boundaries
# ─────────────────────────────────────────────

def find_blocks(cmd_seq):
    """
    cmd_seq: 1D numpy array of command indices (length S)
    Returns: list of (start, end) index pairs — inclusive, one per block
    Each block = SOL ... Ext (all tokens from SOL to matching Ext)
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


# ─────────────────────────────────────────────
#  Level 1 — Token-level masker
# ─────────────────────────────────────────────

class TokenLevelMasker:
    """
    Masks individual curve tokens (Line/Arc/Circle) within sketches.
    SOL and Ext tokens are never masked — they are structural anchors.
    """
    def __init__(self, mask_ratio=0.40):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        """commands: (N, S) long → target_mask: (N, S) bool"""
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np = commands.cpu().numpy()

        for i in range(N):
            curve_pos = [j for j, c in enumerate(cmd_np[i]) if int(c) in SKETCH_CMDS]
            if not curve_pos:
                continue
            n_mask = max(1, int(round(len(curve_pos) * self.mask_ratio)))
            for pos in random.sample(curve_pos, n_mask):
                target_mask[i, pos] = True

        return target_mask


# ─────────────────────────────────────────────
#  Level 2 — Block-level masker (core JEPA)
# ─────────────────────────────────────────────

class OperationBlockMasker:
    """
    Masks complete SOL...Ext operation blocks.
    Always keeps at least one block as context.
    """
    def __init__(self, mask_ratio=0.40):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        """commands: (N, S) long → target_mask: (N, S) bool"""
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np = commands.cpu().numpy()

        for i in range(N):
            blocks = find_blocks(cmd_np[i])
            n_blocks = len(blocks)

            if n_blocks <= 1:
                continue  # nothing to mask without losing all context

            n_mask = max(1, min(int(np.ceil(n_blocks * self.mask_ratio)), n_blocks - 1))
            for idx in random.sample(range(n_blocks), n_mask):
                s, e = blocks[idx]
                target_mask[i, s:e + 1] = True

        return target_mask


# ─────────────────────────────────────────────
#  Level 3 — Group-level masker
# ─────────────────────────────────────────────

class GroupLevelMasker:
    """
    Masks 2-3 consecutive operation blocks as a single group.
    Forces the encoder to understand compound design intent.
    """
    def __init__(self, n_groups=2):
        self.n_groups = n_groups

    def __call__(self, commands):
        """commands: (N, S) long → target_mask: (N, S) bool"""
        N, S = commands.shape
        target_mask = torch.zeros(N, S, dtype=torch.bool, device=commands.device)
        cmd_np = commands.cpu().numpy()

        for i in range(N):
            blocks = find_blocks(cmd_np[i])
            n_blocks = len(blocks)

            if n_blocks <= self.n_groups:
                # Fallback: mask a single block
                if n_blocks > 1:
                    s, e = blocks[random.randint(0, n_blocks - 1)]
                    target_mask[i, s:e + 1] = True
                continue

            # Pick random start, mask n_groups consecutive blocks
            start_b = random.randint(0, n_blocks - self.n_groups)
            for b in range(start_b, start_b + self.n_groups):
                s, e = blocks[b]
                target_mask[i, s:e + 1] = True

        return target_mask


# ─────────────────────────────────────────────
#  Hierarchical masker (randomly selects level)
# ─────────────────────────────────────────────

class HierarchicalMasker:
    """
    Randomly selects masking level per batch step.
    Returns (level_name, target_mask).
    """
    def __init__(self, level_probs=None, n_mask_groups=2):
        self.levels = ['token', 'block', 'group']
        self.probs  = level_probs or [0.33, 0.34, 0.33]
        self.maskers = {
            'token': TokenLevelMasker(mask_ratio=0.40),
            'block': OperationBlockMasker(mask_ratio=0.40),
            'group': GroupLevelMasker(n_groups=n_mask_groups),
        }

    def __call__(self, commands):
        level = random.choices(self.levels, weights=self.probs)[0]
        return level, self.maskers[level](commands)


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def get_masker(strategy, cfg):
    if strategy == 'block':
        return OperationBlockMasker(mask_ratio=cfg.mask_ratio)
    elif strategy == 'token':
        return TokenLevelMasker(mask_ratio=cfg.mask_ratio)
    elif strategy == 'group':
        return GroupLevelMasker(n_groups=cfg.n_mask_groups)
    elif strategy == 'hierarchical':
        return HierarchicalMasker(level_probs=cfg.level_probs, n_mask_groups=cfg.n_mask_groups)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")