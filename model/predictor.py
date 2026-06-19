# %%writefile /content/DeepCAD_prashant/model/predictor.py
import copy
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from model.layers.improved_transformer import (
    TransformerEncoderLayerImproved,
    TransformerDecoderLayerImproved
)
from model.layers.transformer import TransformerEncoder


class _SimpleDecoder(nn.Module):
    """
    Minimal decoder that calls TransformerDecoderLayerImproved directly.
    Bypasses DeepCAD's TransformerDecoder which injects memory2=memory2
    that TransformerDecoderLayerImproved does not accept.
    """
    def __init__(self, layer, n_layers, norm):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)]
        )
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(
                out, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(out) if self.norm is not None else out


class JEPAPredictor(nn.Module):
    """Self-attention predictor — kept for T2 ablation only."""
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.1, max_seq_len=60):
        super().__init__()
        self.input_proj  = nn.Linear(d_model, pred_dim)
        pred_layer       = TransformerEncoderLayerImproved(
            pred_dim, n_heads, dim_feedforward, dropout
        )
        self.transformer = TransformerEncoder(
            pred_layer, n_layers, LayerNorm(pred_dim)
        )
        self.output_proj = nn.Linear(pred_dim, d_model)

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(x)
        x = self.transformer(x, mask=None, src_key_padding_mask=key_padding_mask)
        return self.output_proj(x)


class CADJEPAPredictor(nn.Module):
    """
    Cross-attention predictor — strict I-JEPA design.
    Queries  = position-specific learned embeddings (zero content).
    Memory   = context encoder output projected to pred_dim.
    Uses _SimpleDecoder to avoid DeepCAD TransformerDecoder memory2 conflict.
    """
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.0, max_seq_len=60):
        super().__init__()

        self.position_queries = nn.Embedding(max_seq_len, pred_dim)
        nn.init.trunc_normal_(self.position_queries.weight, std=0.02)

        self.context_proj = nn.Linear(d_model, pred_dim)

        decoder_layer = TransformerDecoderLayerImproved(
            d_model=pred_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = _SimpleDecoder(decoder_layer, n_layers, LayerNorm(pred_dim))

        self.output_proj = nn.Linear(pred_dim, d_model)

    def forward(self, ctx_emb_sf, target_positions,
                key_padding_mask=None, n_real=None):
        """
        ctx_emb_sf:       (S, N, d_model) seq-first
        target_positions: (N, max_n) long
        key_padding_mask: (N, S) — True at EOS/padding
        n_real:           (N,) — actual masked count per sequence
        Returns: (max_n, N, d_model)
        """
        N, max_n = target_positions.shape

        if n_real is not None:
            query_pad = (
                torch.arange(max_n, device=target_positions.device)
                .unsqueeze(0) >= n_real.unsqueeze(1)
            )
        else:
            query_pad = None

        memory     = self.context_proj(ctx_emb_sf)             # (S, N, pred_dim)
        queries    = self.position_queries(target_positions)   # (N, max_n, pred_dim)
        queries_sf = queries.permute(1, 0, 2)                  # (max_n, N, pred_dim)

        out_sf = self.decoder(
            tgt=queries_sf,
            memory=memory,
            tgt_key_padding_mask=query_pad,
            memory_key_padding_mask=key_padding_mask,
        )  # (max_n, N, pred_dim)

        return self.output_proj(out_sf)                        # (max_n, N, d_model)


class HierarchicalPredictor(nn.Module):
    """Three independent CADJEPAPredictor heads — one per masking level."""
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.0, max_seq_len=60):
        super().__init__()
        head_kw = dict(
            d_model=d_model, pred_dim=pred_dim, n_layers=n_layers,
            n_heads=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, max_seq_len=max_seq_len
        )
        self.heads = nn.ModuleDict({
            level: CADJEPAPredictor(**head_kw)
            for level in ['token', 'block', 'group']
        })

    def forward(self, ctx_emb_sf, level, target_positions,
                key_padding_mask=None, n_real=None):
        return self.heads[level](
            ctx_emb_sf, target_positions, key_padding_mask, n_real
        )