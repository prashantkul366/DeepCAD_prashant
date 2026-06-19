import copy
import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    """
    EMA copy of JEPAEncoder. No gradients ever.
    Always processes the full unmasked sequence.
    Updated via exponential moving average after each step.
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
        for o_p, t_p in zip(online_encoder.parameters(), self.encoder.parameters()):
            t_p.data.mul_(d).add_(o_p.data, alpha=1.0 - d)

    @torch.no_grad()
    def forward(self, commands, args):
        """Full sequence forward, no masking. Returns (S, N, d_model)."""
        return self.encoder(commands, args, target_mask=None)