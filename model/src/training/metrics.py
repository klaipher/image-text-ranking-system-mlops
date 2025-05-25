"""
Evaluation metrics for image-text retrieval.
"""

import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity


def compute_retrieval_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    recall_at_k: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Compute retrieval metrics for image-text ranking.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]  
        recall_at_k: Tuple of k values for recall@k computation
        
    Returns:
        Dictionary of metrics
    """
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)
    
    # Text-to-image retrieval
    t2i_metrics = _compute_recall_at_k(similarity_matrix, recall_at_k, "t2i")
    
    # Image-to-text retrieval
    i2t_metrics = _compute_recall_at_k(similarity_matrix.T, recall_at_k, "i2t")
    
    # Mean reciprocal rank
    t2i_mrr = _compute_mrr(similarity_matrix)
    i2t_mrr = _compute_mrr(similarity_matrix.T)
    
    # Combine all metrics
    metrics = {
        **t2i_metrics,
        **i2t_metrics,
        "t2i_mrr": t2i_mrr,
        "i2t_mrr": i2t_mrr,
        "mean_recall_1": (t2i_metrics["t2i_recall_1"] + i2t_metrics["i2t_recall_1"]) / 2,
        "mean_recall_5": (t2i_metrics["t2i_recall_5"] + i2t_metrics["i2t_recall_5"]) / 2,
        "mean_recall_10": (t2i_metrics["t2i_recall_10"] + i2t_metrics["i2t_recall_10"]) / 2,
        "mean_mrr": (t2i_mrr + i2t_mrr) / 2
    }
    
    # Add overall recall_k metrics (using text-to-image by default)
    for k in recall_at_k:
        metrics[f"recall_{k}"] = t2i_metrics[f"t2i_recall_{k}"]
    
    return metrics


def _compute_recall_at_k(
    similarity_matrix: np.ndarray,
    recall_at_k: Tuple[int, ...],
    prefix: str
) -> Dict[str, float]:
    """
    Compute recall@k metrics.
    
    Args:
        similarity_matrix: Similarity matrix [N, N]
        recall_at_k: Tuple of k values
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of recall@k metrics
    """
    n_queries = similarity_matrix.shape[0]
    
    # Get rankings (argsort in descending order)
    rankings = np.argsort(-similarity_matrix, axis=1)
    
    metrics = {}
    for k in recall_at_k:
        recall_at_k_sum = 0
        
        for i in range(n_queries):
            # Check if correct image (index i) is in top-k retrieved images
            if i in rankings[i, :k]:
                recall_at_k_sum += 1
        
        recall_at_k_val = recall_at_k_sum / n_queries
        metrics[f"{prefix}_recall_{k}"] = recall_at_k_val
    
    return metrics


def _compute_mrr(similarity_matrix: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        similarity_matrix: Similarity matrix [N, N]
        
    Returns:
        Mean reciprocal rank
    """
    n_queries = similarity_matrix.shape[0]
    
    # Get rankings (argsort in descending order)
    rankings = np.argsort(-similarity_matrix, axis=1)
    
    reciprocal_ranks = []
    for i in range(n_queries):
        # Find rank of correct image (1-indexed)
        rank = np.where(rankings[i] == i)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
    
    return np.mean(reciprocal_ranks)


def compute_similarity_distribution_stats(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistics about the similarity distribution.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        
    Returns:
        Dictionary of similarity statistics
    """
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)
    
    # Positive pairs (diagonal)
    positive_similarities = np.diag(similarity_matrix)
    
    # Negative pairs (off-diagonal)
    mask = np.eye(similarity_matrix.shape[0], dtype=bool)
    negative_similarities = similarity_matrix[~mask]
    
    stats = {
        "positive_sim_mean": np.mean(positive_similarities),
        "positive_sim_std": np.std(positive_similarities),
        "negative_sim_mean": np.mean(negative_similarities),
        "negative_sim_std": np.std(negative_similarities),
        "sim_gap": np.mean(positive_similarities) - np.mean(negative_similarities),
        "overall_sim_mean": np.mean(similarity_matrix),
        "overall_sim_std": np.std(similarity_matrix)
    }
    
    return stats


def evaluate_model_predictions(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_files: List[str],
    captions: List[str],
    top_k: int = 5
) -> List[Dict]:
    """
    Get detailed predictions for analysis.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        image_files: List of image file names
        captions: List of captions
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)
    
    predictions = []
    
    for i in range(len(captions)):
        # Get top-k similar images for this caption
        similarities = similarity_matrix[i]
        top_indices = np.argsort(-similarities)[:top_k]
        
        prediction = {
            "query_caption": captions[i],
            "ground_truth_image": image_files[i],
            "top_predictions": [
                {
                    "image": image_files[idx],
                    "similarity": float(similarities[idx]),
                    "rank": rank + 1,
                    "is_correct": idx == i
                }
                for rank, idx in enumerate(top_indices)
            ],
            "correct_rank": int(np.where(top_indices == i)[0][0] + 1) if i in top_indices else -1
        }
        
        predictions.append(prediction)
    
    return predictions 