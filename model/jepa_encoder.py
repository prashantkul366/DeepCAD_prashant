import torch
import torch.nn as nn
from model.autoencoder import CADEmbedding
from model.layers.improved_transformer import TransformerEncoderLayerImproved
from model.layers.transformer import TransformerEncoder, LayerNorm
from model.model_utils import (
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
)


class JEPAEncoder(nn.Module):
    """
    CAD sequence encoder for JEPA.
    Returns per-token embeddings (S, N, d_model) instead of mean-pooled z.
    Replaces masked positions with a learnable mask embedding.
    """
    def __init__(self, cfg):
        super().__init__()
        self.use_group = cfg.use_group_emb
        self.d_model   = cfg.d_model

        self.embedding = CADEmbedding(cfg, cfg.max_total_len, use_group=self.use_group)

        enc_layer  = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout
        )
        self.encoder = TransformerEncoder(enc_layer, cfg.n_layers, LayerNorm(cfg.d_model))

        # Learned mask token — replaces content at masked positions
        self.mask_embedding = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)

    def forward(self, commands, args, target_mask=None):
        """
        commands:    (N, S)         long, batch-first
        args:        (N, S, n_args) long, batch-first
        target_mask: (N, S)         bool, True at positions to mask (JEPA targets)

        Returns:
            memory: (S, N, d_model)  per-token embeddings, seq-first
        """
        # ── Convert to seq-first for internal transformer ──
        commands_sf = commands.permute(1, 0)      # (S, N)
        args_sf     = args.permute(1, 0, 2)       # (S, N, n_args)

        key_padding_mask = _get_key_padding_mask(commands_sf, seq_dim=0)   # (N, S)
        group_mask = _get_group_mask(commands_sf, seq_dim=0) if self.use_group else None

        # ── Embed ──────────────────────────────────────────
        src = self.embedding(commands_sf, args_sf, group_mask)  # (S, N, d_model)

        # ── Replace target positions with mask embedding ───
        if target_mask is not None:
            mask_sf  = target_mask.permute(1, 0).unsqueeze(-1)  # (S, N, 1)
            mask_emb = self.mask_embedding.view(1, 1, -1).expand_as(src)
            src = torch.where(mask_sf, mask_emb, src)

        # ── Encode ─────────────────────────────────────────
        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)
        return memory  # (S, N, d_model)

    @torch.no_grad()
    def get_pooled_embedding(self, commands, args):
        """
        For downstream evaluation — mean pool over non-padding tokens.
        commands: (N, S), args: (N, S, n_args)
        Returns: (N, d_model)
        """
        commands_sf = commands.permute(1, 0)       # (S, N)
        memory      = self.forward(commands, args)  # (S, N, d_model)
        padding_mask = _get_padding_mask(commands_sf, seq_dim=0)  # (S, N, 1)
        z = (memory * padding_mask).sum(dim=0) / padding_mask.sum(dim=0).clamp(min=1)
        return z  # (N, d_model)