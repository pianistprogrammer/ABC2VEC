"""Evaluation utilities for ABC2Vec."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from typing import Dict, List, Tuple


def compute_similarity_matrix(
    embeddings: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Embedding matrix (num_samples, d_embed)
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Similarity matrix (num_samples, num_samples)
    """
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

    return embeddings @ embeddings.T


def evaluate_retrieval_at_k(
    similarities: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
    exclude_self: bool = True,
) -> Dict[str, float]:
    """
    Evaluate retrieval accuracy at different k values.

    For each query, finds top-k nearest neighbors and checks
    if any match the query's label.

    Args:
        similarities: Similarity matrix (num_queries, num_candidates)
        labels: Label array (num_samples,)
        k_values: List of k values to evaluate
        exclude_self: Whether to exclude self from neighbors

    Returns:
        Dictionary with Recall@k metrics
    """
    num_samples = similarities.shape[0]
    metrics = {}

    for k in k_values:
        correct = 0

        for i in range(num_samples):
            sims = similarities[i].copy()

            # Exclude self-similarity if needed
            if exclude_self:
                sims[i] = -np.inf

            # Get top-k indices
            top_k_indices = np.argsort(sims)[-k:]

            # Check if any neighbor has same label
            query_label = labels[i]
            neighbor_labels = labels[top_k_indices]

            if query_label in neighbor_labels and query_label != "":
                correct += 1

        recall = correct / num_samples
        metrics[f"recall@{k}"] = recall

    return metrics


def mean_average_precision_at_k(
    similarities: np.ndarray,
    labels: np.ndarray,
    k: int = 100,
    exclude_self: bool = True,
) -> float:
    """
    Compute Mean Average Precision at k (MAP@k).

    Args:
        similarities: Similarity matrix (num_queries, num_candidates)
        labels: Label array (num_samples,)
        k: Number of top results to consider
        exclude_self: Whether to exclude self from neighbors

    Returns:
        MAP@k score
    """
    num_samples = similarities.shape[0]
    average_precisions = []

    for i in range(num_samples):
        sims = similarities[i].copy()

        if exclude_self:
            sims[i] = -np.inf

        # Get top-k indices
        top_k_indices = np.argsort(sims)[-k:][::-1]  # Descending order

        query_label = labels[i]
        if query_label == "":
            continue

        # Compute precision at each position
        precisions = []
        num_relevant_found = 0

        for rank, idx in enumerate(top_k_indices, start=1):
            if labels[idx] == query_label:
                num_relevant_found += 1
                precision_at_rank = num_relevant_found / rank
                precisions.append(precision_at_rank)

        if precisions:
            average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions) if average_precisions else 0.0


def mean_reciprocal_rank(
    similarities: np.ndarray,
    labels: np.ndarray,
    exclude_self: bool = True,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR measures the rank of the first relevant result.

    Args:
        similarities: Similarity matrix
        labels: Label array
        exclude_self: Whether to exclude self

    Returns:
        MRR score
    """
    num_samples = similarities.shape[0]
    reciprocal_ranks = []

    for i in range(num_samples):
        sims = similarities[i].copy()

        if exclude_self:
            sims[i] = -np.inf

        # Get sorted indices (descending similarity)
        sorted_indices = np.argsort(sims)[::-1]

        query_label = labels[i]
        if query_label == "":
            continue

        # Find rank of first relevant result
        for rank, idx in enumerate(sorted_indices, start=1):
            if labels[idx] == query_label:
                reciprocal_ranks.append(1.0 / rank)
                break

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
