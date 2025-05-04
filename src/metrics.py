import numpy as np

def _dcg(rels):
    return np.sum((2 ** rels - 1) / np.log2(np.arange(2, rels.size + 2)))

def ndcg(relevance, scores, k=10):
    idx = np.argsort(scores)[::-1][:k]
    gains = relevance[idx]
    dcg  = _dcg(gains)
    ideal = _dcg(np.sort(relevance)[::-1][:k])
    return dcg / ideal if ideal > 0 else 0.0

def mrr(relevance, scores, k=10):
    idx = np.argsort(scores)[::-1][:k]
    rels = relevance[idx]
    hits = np.where(rels > 0)[0]
    return 1 / (hits[0] + 1) if hits.size else 0.0

def recall_at_k(relevance, scores, k=10):
    idx = np.argsort(scores)[::-1][:k]
    hits = (relevance[idx] > 0).sum()
    total = (relevance > 0).sum()
    return hits / total if total else 0.0