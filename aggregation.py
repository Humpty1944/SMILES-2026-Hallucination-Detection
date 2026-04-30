"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations
import numpy as np
import torch


TARGET_LAYERS = [12, 13, 14, 15, 16, ]
LAST_TOKENS_COUNT = 3

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero().squeeze()
    
    if real_positions.numel()==0:
        return hidden_states[0][0]*0

    features=[]

    for layer_idx in TARGET_LAYERS:
        layer_states =hidden_states[layer_idx]

        # last token with the most info
        last_token = layer_states[real_positions[-1]]
        features.append(last_token)

        # mean pooling last LAST_TOKENS_COUNT tokens for stabilaizing
        if len(real_positions)>=LAST_TOKENS_COUNT:
            last_n_tokens = layer_states[real_positions[-LAST_TOKENS_COUNT:]]
            mean_pool= last_n_tokens.mean(dim=0)
            features.append(mean_pool)
        else:
            # if less than LAST_TOKENS_COUNT take mean
            mean_pool = layer_states[real_positions].mean(dim=0)
            features.append(mean_pool)

    stacked = torch.stack(features, dim=0)
    pooled = stacked.mean(dim=0) 

    return pooled
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero().squeeze()
    last_token_idx = real_positions[-1]

    # for geometric takes last tokens for every TARGET_LAYERS
    tokens = torch.stack([
        hidden_states[layer_idx][last_token_idx]
        for layer_idx in TARGET_LAYERS
    ], dim=0)

    features=[]
    features.append(torch.pdist(tokens, p=2).mean().item() if tokens.size(0)>1 else 0.0) # mean dist between all vectors
    features.append(tokens.mean(dim=0).norm().item()) #len of the mean vector

    return torch.tensor(features, device = tokens.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
