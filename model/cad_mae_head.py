import torch
import torch.nn as nn
import torch.nn.functional as F

class CADMAEHead(nn.Module):
    """
    Reconstruction head for CAD masked autoencoding.
    Given encoder output at masked positions,
    predicts command type (6-class) and argument values (16 × 256-class).
    """
    def __init__(self, d_model, n_commands=6, n_args=16, args_dim=256):
        super().__init__()
        self.n_args   = n_args
        self.args_dim = args_dim
        self.cmd_head  = nn.Linear(d_model, n_commands)
        self.args_head = nn.Linear(d_model, n_args * args_dim)

    def forward(self, memory_sf, target_mask):
        """
        memory_sf:   (S, N, d)  encoder output, seq-first
        target_mask: (N, S)     bool, True = masked position
        Returns:
            cmd_logits  (M, n_commands)
            args_logits (M, n_args, args_dim)
        where M = total masked tokens across batch
        """
        mask_sf  = target_mask.permute(1, 0)   # (S, N)
        flat_emb = memory_sf[mask_sf]           # (M, d)

        cmd_logits  = self.cmd_head(flat_emb)
        args_logits = self.args_head(flat_emb).view(-1, self.n_args, self.args_dim)
        return cmd_logits, args_logits


def mae_loss(cmd_logits, args_logits, commands, args, target_mask, pad_val=-1):
    """
    cmd_logits:  (M, 6)
    args_logits: (M, 16, 256)
    commands:    (N, S)
    args:        (N, S, 16)
    target_mask: (N, S) bool
    """
    mask_sf  = target_mask.permute(1, 0)             # (S, N)
    cmd_true = commands.permute(1, 0)[mask_sf]        # (M,)
    arg_true = args.permute(1, 0, 2)[mask_sf]         # (M, 16)

    loss_cmd = F.cross_entropy(cmd_logits, cmd_true)

    valid = (arg_true != pad_val)                     # (M, 16)
    if valid.sum() > 0:
        arg_true_safe = arg_true.clone()
        arg_true_safe[~valid] = 0
        loss_args = F.cross_entropy(
            args_logits.reshape(-1, 256)[valid.reshape(-1)],
            arg_true_safe.reshape(-1)[valid.reshape(-1)]
        )
    else:
        loss_args = torch.tensor(0.0, device=cmd_logits.device)

    return loss_cmd + loss_args