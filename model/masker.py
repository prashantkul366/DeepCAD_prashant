# %%writefile /content/DeepCAD_prashant/model/masker.py
import random
import torch
import numpy as np
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX

SKETCH_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}

# Alias for backward compatibility
# HierarchicalMasker = AdaptiveMasker

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
    """
    Mask individual curve tokens (Line/Arc/Circle).
    SOL, Ext, EOS are permanent context — never masked.
    Works on ALL sequences including single-curve sketches.
    Context for 1-curve seqs = SOL + Ext (geometric anchor, sufficient).
    """
    curve_pos = [j for j, c in enumerate(cmd_np_i) if int(c) in SKETCH_CMDS]
    if not curve_pos:
        return   # no curve tokens at all — skip (should not occur in valid data)
    n_mask = max(1, int(round(len(curve_pos) * mask_ratio)))
    n_mask = min(n_mask, len(curve_pos))  # never exceed available curves
    for pos in random.sample(curve_pos, n_mask):
        mask_i[pos] = True


# def _apply_block_masking(blocks, mask_i, mask_ratio=0.40):
#     """
#     Mask complete SOL...Ext blocks. Keeps at least one block as context.
#     Caller guarantees len(blocks) >= 2.
#     """
#     n_blocks = len(blocks)
#     n_mask   = max(1, min(int(np.ceil(n_blocks * mask_ratio)), n_blocks - 1))
#     for idx in random.sample(range(n_blocks), n_mask):
#         s, e = blocks[idx]
#         mask_i[s:e + 1] = True

def _apply_block_masking(blocks, mask_i, mask_ratio=0.40, n_targets=1):
    """
    Mask complete SOL...Ext blocks. Keeps at least one block as context.
    n_targets > 1: multi-target mode — mask exactly min(n_targets, n_blocks-1) blocks.
    """
    n_blocks = len(blocks)
    if n_targets > 1:
        n_mask = max(1, min(n_targets, n_blocks - 1))
    else:
        n_mask = max(1, min(int(np.ceil(n_blocks * mask_ratio)), n_blocks - 1))
    for idx in random.sample(range(n_blocks), n_mask):
        s, e = blocks[idx]
        mask_i[s:e + 1] = True


def _apply_group_masking(blocks, mask_i, n_groups=2):
    """
    Mask n_groups consecutive blocks.
    Caller guarantees len(blocks) >= n_groups + 1.
    """
    n_blocks = len(blocks)
    start_b  = random.randint(0, n_blocks - n_groups)
    for b in range(start_b, start_b + n_groups):
        s, e = blocks[b]
        mask_i[s:e + 1] = True


class TokenLevelMasker:
    """Pure token masking. Works on all sequences."""
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
    """
    Block masking with token fallback for single-block sequences.
    Single-block → token masking (block masking impossible without losing all context)
    """
    def __init__(self, mask_ratio=0.40, mask_ratio_token=0.50, n_targets=1):
        self.mask_ratio       = mask_ratio
        self.mask_ratio_token = mask_ratio_token
        self.n_targets        = n_targets

    def __call__(self, commands):
        N, S    = commands.shape
        mask_np = np.zeros((N, S), dtype=bool)
        cmd_np  = commands.cpu().numpy()
        for i in range(N):
            blocks = find_blocks(cmd_np[i])
            if len(blocks) >= 2:
                _apply_block_masking(blocks, mask_np[i], self.mask_ratio, self.n_targets)
            else:
                _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio_token)
        return torch.from_numpy(mask_np).to(commands.device)


class GroupLevelMasker:
    """
    Group masking with graceful fallbacks:
      n_blocks >= 3 : group masking
      n_blocks == 2 : block masking
      n_blocks == 1 : token masking
    """
    def __init__(self, n_groups=2, mask_ratio=0.40, mask_ratio_token=0.50):
        self.n_groups         = n_groups
        self.mask_ratio       = mask_ratio
        self.mask_ratio_token = mask_ratio_token

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
                _apply_block_masking(blocks, mask_np[i], self.mask_ratio)
            else:
                _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio_token)
        return torch.from_numpy(mask_np).to(commands.device)


class AdaptiveMasker:
    """
    Per-sequence level selection based on block count:
      1 block  → token only (60.4% of dataset)
      2 blocks → token(50%) or block(50%)
      3+blocks → token(25%), block(50%), group(25%)

    Returns (target_mask [N,S bool], level_per_seq [list of str])
    """
    def __init__(self, mask_ratio=0.40, mask_ratio_token=0.50, n_groups=2, n_targets=1,
                 use_curriculum=False):
        self.mask_ratio       = mask_ratio
        self.mask_ratio_token = mask_ratio_token
        self.n_groups         = n_groups
        self.n_targets        = n_targets
        self.use_curriculum   = use_curriculum

    def _get_level_weights(self, epoch):
        """
        Curriculum schedule:
          ep 0-29:  token only    → rank rises fast via easy predictions
          ep 30-79: token → block → encoder adapts to operation-level masking
          ep 80+:   full hierarchical (standard weights)
        """
        if not self.use_curriculum or epoch >= 80:
            return [0.25, 0.50, 0.25]
        elif epoch < 30:
            return [1.00, 0.00, 0.00]
        else:
            t = (epoch - 30) / 50.0   # 0→1 over ep30-79
            return [1.0 - 0.75*t, 0.75*t, 0.0]

    # def __call__(self, commands):
    def __call__(self, commands, epoch=0):
        N, S          = commands.shape
        mask_np       = np.zeros((N, S), dtype=bool)
        cmd_np        = commands.cpu().numpy()
        level_per_seq = []

        for i in range(N):
            blocks   = find_blocks(cmd_np[i])
            n_blocks = len(blocks)

            # Select level per sequence
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

            # Apply masking
            if level == 'token':
                _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio_token)
            elif level == 'block':
                if n_blocks >= 2:
                    _apply_block_masking(blocks, mask_np[i], self.mask_ratio, self.n_targets)
                else:
                    _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio_token)
            elif level == 'group':
                if n_blocks >= self.n_groups + 1:
                    _apply_group_masking(blocks, mask_np[i], self.n_groups)
                elif n_blocks >= 2:
                    _apply_block_masking(blocks, mask_np[i], self.mask_ratio, self.n_targets)
                else:
                    _apply_token_masking(cmd_np[i], mask_np[i], self.mask_ratio_token)

        target_mask = torch.from_numpy(mask_np).to(commands.device)
        return target_mask, level_per_seq


def get_masker(strategy, cfg):
    tok_ratio = getattr(cfg, 'mask_ratio_token', 0.50)
    blk_ratio = getattr(cfg, 'mask_ratio',       0.40)
    n_groups  = getattr(cfg, 'n_mask_groups',     2)

    n_targets = getattr(cfg, 'n_mask_targets', 1)

    if strategy == 'token':
        return TokenLevelMasker(mask_ratio=tok_ratio)
    elif strategy == 'block':
        return OperationBlockMasker(
            mask_ratio=blk_ratio, mask_ratio_token=tok_ratio,
            n_targets=n_targets
        )
    elif strategy == 'group':
        return GroupLevelMasker(
            n_groups=n_groups, mask_ratio=blk_ratio,
            mask_ratio_token=tok_ratio
        )
    elif strategy == 'hierarchical':
        return AdaptiveMasker(
            mask_ratio=blk_ratio, mask_ratio_token=tok_ratio,
            n_groups=n_groups, n_targets=n_targets,
            use_curriculum=getattr(cfg, 'curriculum', False)
        )
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
    

# Alias for backward compatibility
HierarchicalMasker = AdaptiveMasker