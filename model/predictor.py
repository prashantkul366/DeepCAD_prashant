import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from model.layers.improved_transformer import (
    TransformerEncoderLayerImproved,
    TransformerDecoderLayerImproved
)
from model.layers.transformer import TransformerEncoder, TransformerDecoder


class JEPAPredictor(nn.Module):
    """
    Self-attention predictor.
    Kept only for T2 ablation (masking strategy comparison).
    Weaker than CADJEPAPredictor: encoder already contextualizes masked positions
    via self-attention before this predictor sees them.
    """
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.1):
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
        """x: (S, N, d_model) → (S, N, d_model)"""
        x = self.input_proj(x)
        x = self.transformer(x, mask=None, src_key_padding_mask=key_padding_mask)
        return self.output_proj(x)


class CADJEPAPredictor(nn.Module):
    """
    Cross-attention predictor following I-JEPA design.

    Architecture:
      Queries  = position-specific learned tokens at target positions
                 → carry ONLY positional information, zero content
      Keys/Values = context encoder output projected to pred_dim
                    → the only source of semantic information

    Why this forces good representations:
      The predictor cannot shortcut by reading content from its queries.
      The context encoder must encode rich inter-operation geometric context
      into its output for the cross-attention to extract and predict from.
      This forces block-level and operation-level semantic structure into
      the encoder representations — which is exactly what CAD understanding
      downstream tasks (retrieval, probing, classification) require.

    Target positions are padded to max_n per batch.
    Loss is computed only at real positions using n_real per sequence.
    """
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.0, max_seq_len=60):
        super().__init__()

        # Position-specific query tokens: one per possible sequence position.
        # These are the ONLY information the predictor receives about target
        # positions — zero content. Initialized small, learns position patterns.
        self.position_queries = nn.Embedding(max_seq_len, pred_dim)
        nn.init.trunc_normal_(self.position_queries.weight, std=0.02)

        # Project context encoder output to narrower predictor dim.
        # Narrowing (pred_dim < d_model) prevents the predictor from
        # trivially copying encoder representations.
        self.context_proj = nn.Linear(d_model, pred_dim)

        # Cross-attention decoder: queries attend to projected context.
        decoder_layer = TransformerDecoderLayerImproved(
            d_model=pred_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            decoder_layer, n_layers, LayerNorm(pred_dim)
        )

        # Project predictions back to encoder dim for loss computation.
        self.output_proj = nn.Linear(pred_dim, d_model)

    def forward(self, ctx_emb_sf, target_positions,
                key_padding_mask=None, n_real=None):
        """
        ctx_emb_sf:       (S, N, d_model) — context encoder output, seq-first
        target_positions: (N, max_n)      — padded masked position indices (long)
        key_padding_mask: (N, S)          — True at EOS/padding positions
        n_real:           (N,)            — actual masked count per sequence
                                            used to mask padded queries

        Returns: (max_n, N, d_model) — predictions at target positions, seq-first
        """
        N, max_n = target_positions.shape

        # Query padding mask — prevents padded queries from attending to each other
        # in the decoder self-attention layers.
        if n_real is not None:
            query_pad = (
                torch.arange(max_n, device=target_positions.device)
                .unsqueeze(0) >= n_real.unsqueeze(1)
            )  # (N, max_n) — True at padded positions
        else:
            query_pad = None

        # Project context to predictor dim: (S, N, pred_dim)
        memory = self.context_proj(ctx_emb_sf)

        # Position queries — zero content, only position:
        # (N, max_n) → (N, max_n, pred_dim) → (max_n, N, pred_dim)
        queries    = self.position_queries(target_positions)
        queries_sf = queries.permute(1, 0, 2)

        # Cross-attend: queries extract from context
        # out_sf = self.decoder(
        #     tgt=queries_sf,                        # (max_n, N, pred_dim)
        #     memory=memory,                         # (S,     N, pred_dim)
        #     tgt_mask=None,
        #     tgt_key_padding_mask=query_pad,        # (N, max_n)
        # )  # (max_n, N, pred_dim)

        out_sf = self.decoder(
            tgt=queries_sf,
            memory=memory,
            tgt_mask=None,
            tgt_key_padding_mask=query_pad,
            memory_key_padding_mask=key_padding_mask,
        )

        return self.output_proj(out_sf)            # (max_n, N, d_model)


class HierarchicalPredictor(nn.Module):
    """
    Three independent CADJEPAPredictor heads — one per masking level.
    Each head specialises on the prediction task at its granularity:
      token head  → predict individual curve parameter embeddings
      block head  → predict operation-level embeddings from sibling operations
      group head  → predict compound design-intent embeddings
    """
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
        """Route to correct head. Returns (max_n, N, d_model)."""
        return self.heads[level](
            ctx_emb_sf, target_positions, key_padding_mask, n_real
        )