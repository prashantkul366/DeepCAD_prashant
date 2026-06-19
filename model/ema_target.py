# %%writefile /content/DeepCAD_prashant/model/ema_target.py
import copy
import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    """
    Exponential moving average copy of JEPAEncoder.
    Never receives gradients — updated only via .update() after each step.
    Always sees the full unmasked sequence.
    """
    def __init__(self, online_encoder, decay=0.996):
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        self.decay   = decay
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder, decay=None):
        d = decay if decay is not None else self.decay
        for o_p, t_p in zip(
            online_encoder.parameters(),
            self.encoder.parameters()
        ):
            t_p.data.mul_(d).add_(o_p.data, alpha=1.0 - d)

    @torch.no_grad()
    def forward(self, commands, args):
        """Full sequence, no masking. Returns (S, N, d_model)."""
        return self.encoder(commands, args, target_mask=None)