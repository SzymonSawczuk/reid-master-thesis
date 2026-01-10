"""
Matching and Ranking Module

Handles feature matching and ranking with optional reranking.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from .reranking import re_ranking


class MatchingModule:
    """
    Matching and ranking module with reranking support.
    """

    def __init__(
        self,
        use_reranking: bool = True,
        k1: int = 20,
        k2: int = 6,
        lambda_value: float = 0.3,
        distance_metric: str = 'euclidean'
    ):
        """
        Initialize matching module.

        Args:
            use_reranking: Whether to use reranking
            k1: k1 parameter for reranking
            k2: k2 parameter for reranking
            lambda_value: lambda parameter for reranking
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        self.use_reranking = use_reranking
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        self.distance_metric = distance_metric

        print(f"Matching module initialized")
        print(f"  Reranking: {use_reranking}")
        print(f"  Distance metric: {distance_metric}")

    def compute_distance_matrix(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        use_reranking: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute distance matrix between query and gallery features.

        Args:
            query_features: Query feature vectors (shape: [N, D])
            gallery_features: Gallery feature vectors (shape: [M, D])
            use_reranking: Whether to use reranking (overrides default)

        Returns:
            Distance matrix (shape: [N, M])
        """
        if use_reranking is None:
            use_reranking = self.use_reranking

        if self.distance_metric == 'cosine':
            
            distmat = 1 - np.dot(query_features, gallery_features.T)
        else:
            
            distmat = self._compute_euclidean_distance(
                query_features,
                gallery_features
            )

        if use_reranking:
            
            distmat = re_ranking(
                query_features,
                gallery_features,
                k1=self.k1,
                k2=self.k2,
                lambda_value=self.lambda_value
            )

        return distmat

    def _compute_euclidean_distance(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute Euclidean distance matrix using torch.cdist for consistency with torchreid.

        Args:
            query_features: Query features
            gallery_features: Gallery features

        Returns:
            Distance matrix
        """
        
        q_feats = torch.from_numpy(query_features).float()
        g_feats = torch.from_numpy(gallery_features).float()
        distmat = torch.cdist(q_feats, g_feats).numpy()
        return distmat

    def match_batch(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a batch of queries against gallery.

        Args:
            query_features: Query feature vectors (shape: [N, D])
            gallery_features: Gallery feature vectors (shape: [M, D])
            top_k: Number of top matches to return per query

        Returns:
            all_indices: Indices of top matches (shape: [N, top_k])
            all_distances: Distances of top matches (shape: [N, top_k])
        """
        
        distmat = self.compute_distance_matrix(
            query_features,
            gallery_features
        )

        
        all_indices = np.argsort(distmat, axis=1)[:, :top_k]
        all_distances = np.take_along_axis(distmat, all_indices, axis=1)

        return all_indices, all_distances

