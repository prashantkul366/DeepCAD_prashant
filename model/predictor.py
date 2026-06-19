# %%writefile /content/DeepCAD_prashant/model/predictor.py
import torch.nn as nn
from model.layers.improved_transformer import TransformerEncoderLayerImproved
from model.layers.transformer import TransformerEncoder, LayerNorm


class JEPAPredictor(nn.Module):
    """
    Narrow predictor transformer.
    Input:  context encoder output (S, N, d_model)
    Output: predicted embeddings at all positions (S, N, d_model)
    Loss is computed externally only at target (masked) positions.

    Kept narrow (pred_dim < d_model) so the predictor cannot memorize
    without the encoder building useful representations.
    """
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(d_model, pred_dim)
        pred_layer = TransformerEncoderLayerImproved(
            pred_dim, n_heads, dim_feedforward, dropout
        )
        self.transformer = TransformerEncoder(
            pred_layer, n_layers, LayerNorm(pred_dim)
        )
        self.output_proj = nn.Linear(pred_dim, d_model)

    def forward(self, x, key_padding_mask=None):
        """
        x: (S, N, d_model)
        Returns: (S, N, d_model)
        """
        x = self.input_proj(x)
        x = self.transformer(x, mask=None, src_key_padding_mask=key_padding_mask)
        x = self.output_proj(x)
        return x


class HierarchicalPredictor(nn.Module):
    """
    Three independent predictor heads — one per masking level.
    Each head is a JEPAPredictor with the same architecture.
    """
    def __init__(self, d_model, pred_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0.1):
        super().__init__()
        head_kwargs = dict(
            d_model=d_model, pred_dim=pred_dim, n_layers=n_layers,
            n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.heads = nn.ModuleDict({
            level: JEPAPredictor(**head_kwargs)
            for level in ['token', 'block', 'group']
        })

    def forward(self, x, level, key_padding_mask=None):
        """Route to the correct head for this masking level."""
        return self.heads[level](x, key_padding_mask)