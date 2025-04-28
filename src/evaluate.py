# src/evaluate.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from src.metrics import compute_ndcg, compute_mrr
from src.helper import clean_text, load_and_preprocess

def evaluate(df, model_name='sentence-transformers/multi-qa-mpnet-base-cos-v1', batch_size=32, device=None):
    """Evaluate model on queries and titles."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SentenceTransformer(model_name).to(device)

    ndcg_scores = []
    mrr_scores = []
    queries = []

    grouped = df.groupby('query')

    for query, group in tqdm(grouped, desc="Evaluating queries"):
        queries.append(query)

        query_vec = model.encode(query, convert_to_tensor=True, device=device)
        titles = group['title'].tolist()
        title_vecs = model.encode(titles, batch_size=batch_size, convert_to_tensor=True, device=device)

        scores = util.cos_sim(query_vec, title_vecs)[0]
        relevances = torch.tensor(group['relevance'].to_numpy(), device=device)

        scores_np = scores.detach().cpu().numpy()
        relevances_np = relevances.detach().cpu().numpy()

        ndcg = compute_ndcg(relevances_np, scores_np, k=10)
        mrr = compute_mrr(relevances_np, scores_np, k=10)

        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)

    # Build DataFrame
    results = pd.DataFrame({
        'query': queries,
        'NDCG@10': ndcg_scores,
        'MRR@10': mrr_scores
    })

    return results