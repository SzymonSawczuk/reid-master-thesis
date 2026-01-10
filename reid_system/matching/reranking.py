"""
Re-ranking algorithm for person/vehicle re-identification.

Based on: Zhong et al. "Re-ranking Person Re-identification with k-reciprocal Encoding"
Uses torchreid's re-ranking implementation for consistency with evaluation.
"""

import torch
import numpy as np
import torchreid


def re_ranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3
) -> np.ndarray:
    """
    Re-ranking algorithm using k-reciprocal encoding via torchreid.

    Args:
        query_features: Query feature vectors (shape: [N_q, D])
        gallery_features: Gallery feature vectors (shape: [N_g, D])
        k1: k1 parameter for reciprocal neighbors
        k2: k2 parameter for local query expansion
        lambda_value: Weight balance parameter

    Returns:
        Re-ranked distance matrix (shape: [N_q, N_g])
    """
    
    q_feats = torch.from_numpy(query_features).float()
    g_feats = torch.from_numpy(gallery_features).float()

    distmat = torchreid.metrics.compute_distance_matrix(q_feats, g_feats, metric='euclidean').numpy()
    distmat_qq = torchreid.metrics.compute_distance_matrix(q_feats, q_feats, metric='euclidean').numpy()
    distmat_gg = torchreid.metrics.compute_distance_matrix(g_feats, g_feats, metric='euclidean').numpy()

    
    distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg, k1=k1, k2=k2, lambda_value=lambda_value)

    
    if isinstance(distmat, torch.Tensor):
        distmat = distmat.numpy()

    return distmat


def compute_euclidean_distance(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance matrix using torch.cdist.

    Args:
        features1: First feature matrix (shape: [N, D])
        features2: Second feature matrix (shape: [M, D])

    Returns:
        Distance matrix (shape: [N, M])
    """
    feat1_torch = torch.from_numpy(features1).float()
    feat2_torch = torch.from_numpy(features2).float()
    distmat = torch.cdist(feat1_torch, feat2_torch).numpy()
    return distmat
