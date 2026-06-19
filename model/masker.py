# %%writefile /content/DeepCAD_prashant/model/masker.py
import random
import torch
import numpy as np
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX

SKETCH_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}


def find_blocks(cmd_seq):
    """
    cmd_seq: 1D numpy array of command indices
    Returns list of (start, end) inclusive — one per SOL...Ext block
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


def _apply_token_masking(cmd_np_i, mask_i, mask_ratio=0.50):
    """Mask individual curve tokens within sketches. Never masks SOL/Ext/EOS."""
    curve_pos = [j for j, c in enumerate(cmd_np_i) if int(c) in SKETCH_CMDS]
    if len(curve_pos) < 2:
        return   # skip degenerate single-curve sketches
    n_mask = max(1, int(round(len(curve_pos) * mask_ratio)))
    for pos in random.sample(curve_pos, min(n_mask, len(curve_pos))):
        mask_i[pos] = True


def _apply_block_masking(blocks, mask_i, mask_ratio=0.40):
    """Mask complete SOL...Ext blocks. Keeps at least one block as context."""
    n_blocks = len(blocks)
    n_mask = max(1, min(int(np.ceil(n_blocks * mask_ratio)), n_blocks - 1))
    for idx in random.sample(range(n_blocks), n_mask):
        s, e = blocks[idx]
        mask_i[s:e + 1] = True


def _apply_group_masking(blocks, mask_i, n_groups=2):
    """Mask n_groups consecutive blocks. Caller ensures len(blocks) >= n_groups+1."""
    n_blocks = len(blocks)
    start_b  = random.randint(0, n_blocks - n_groups)
    for b in range(start_b, start_b + n_groups):
        s, e = blocks[b]
        mask_i[s:e + 1] = True


class TokenLevelMasker:
    def __init__(self, mask_ratio=0.50):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        for i in range(N):
            _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio)
        return torch.from_numpy(mask_np).to(commands.device)


class OperationBlockMasker:
    """Block masking with token fallback for single-block sequences."""
    def __init__(self, mask_ratio=0.40):
        self.mask_ratio = mask_ratio

    def __call__(self, commands):
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        for i in range(N):
            blocks = find_blocks(cmd_np[i])
            if len(blocks) >= 2:
                _apply_block_masking(blocks, mask_np[i], self.mask_ratio)
            else:
                _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.50)
        return torch.from_numpy(mask_np).to(commands.device)


class GroupLevelMasker:
    """Group masking with graceful fallbacks for short sequences."""
    def __init__(self, n_groups=2):
        self.n_groups = n_groups

    def __call__(self, commands):
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        for i in range(N):
            blocks   = find_blocks(cmd_np[i])
            n_blocks = len(blocks)
            if n_blocks >= self.n_groups + 1:
                _apply_group_masking(blocks, mask_np[i], self.n_groups)
            elif n_blocks >= 2:
                _apply_block_masking(blocks, mask_np[i], mask_ratio=0.40)
            else:
                _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.50)
        return torch.from_numpy(mask_np).to(commands.device)


class AdaptiveMasker:
    """
    Selects masking level per sequence based on block count.
    Ensures every shape contributes gradient signal:
      1 block  → token only
      2 blocks → token(50%) or block(50%)
      3+blocks → token(25%), block(50%), group(25%)
    Returns (target_mask [N,S bool], level_per_seq [list of str])
    """
    def __init__(self, mask_ratio=0.40, mask_ratio_token=0.50, n_groups=2):
        self.maskers = {
            'token': TokenLevelMasker(mask_ratio=mask_ratio_token),
            'block': OperationBlockMasker(mask_ratio=mask_ratio),
            'group': GroupLevelMasker(n_groups=n_groups),
        }
        self.n_groups = n_groups

    def __call__(self, commands):
        N, S           = commands.shape
        mask_np        = np.zeros((N, S), dtype=bool)
        cmd_np         = commands.cpu().numpy()
        level_per_seq  = []

        for i in range(N):
            blocks   = find_blocks(cmd_np[i])
            n_blocks = len(blocks)

            if n_blocks == 1:
                level = 'token'
            elif n_blocks == 2:
                level = random.choice(['token', 'block'])
            else:
                level = random.choices(
                    ['token', 'block', 'group'],
                    weights=[0.25, 0.50, 0.25]
                )[0]

            level_per_seq.append(level)

            if level == 'token':
                _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.50)
            elif level == 'block':
                if len(blocks) >= 2:
                    _apply_block_masking(blocks, mask_np[i], mask_ratio=0.40)
                else:
                    _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.50)
            elif level == 'group':
                if len(blocks) >= self.n_groups + 1:
                    _apply_group_masking(blocks, mask_np[i], self.n_groups)
                elif len(blocks) >= 2:
                    _apply_block_masking(blocks, mask_np[i], mask_ratio=0.40)
                else:
                    _apply_token_masking(cmd_np[i], mask_np[i], mask_ratio=0.50)

        target_mask = torch.from_numpy(mask_np).to(commands.device)
        return target_mask, level_per_seq


def get_masker(strategy, cfg):
    tok_ratio = getattr(cfg, 'mask_ratio_token', 0.50)
    blk_ratio = getattr(cfg, 'mask_ratio', 0.40)
    n_groups  = getattr(cfg, 'n_mask_groups', 2)

    if strategy == 'token':
        return TokenLevelMasker(mask_ratio=tok_ratio)
    elif strategy == 'block':
        return OperationBlockMasker(mask_ratio=blk_ratio)
    elif strategy == 'group':
        return GroupLevelMasker(n_groups=n_groups)
    elif strategy == 'hierarchical':
        return AdaptiveMasker(
            mask_ratio=blk_ratio,
            mask_ratio_token=tok_ratio,
            n_groups=n_groups
        )
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")