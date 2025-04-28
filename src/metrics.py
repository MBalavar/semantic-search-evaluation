# src/metrics.py

import numpy as np

def compute_ndcg(relevances, scores, k=10):
    """Compute Normalized Discounted Cumulative Gain at K."""
    order = np.argsort(scores)[::-1][:k]
    gains = np.take(relevances, order)
    discounts = np.log2(np.arange(2, gains.size + 2))
    dcg = np.sum(gains / discounts)
    
    ideal_order = np.argsort(relevances)[::-1][:k]
    ideal_gains = np.take(relevances, ideal_order)
    ideal_dcg = np.sum(ideal_gains / discounts)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_mrr(relevances, scores, k=10):
    """Compute Mean Reciprocal Rank at K."""
    order = np.argsort(scores)[::-1][:k]
    for idx, i in enumerate(order):
        if relevances[i] > 0:
            return 1.0 / (idx + 1)
    return 0.0