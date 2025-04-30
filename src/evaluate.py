# src/evaluate.py

import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from src.metrics import compute_ndcg, compute_mrr
from src.helper import clean_text, load_and_preprocess

def evaluate(df, model_name='sentence-transformers/multi-qa-mpnet-base-cos-v1', batch_size=32, device=None):
    """Evaluate model on queries and titles."""
    
    start_time = time.time()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_load_start = time.time()
    model = SentenceTransformer(model_name).to(device)
    model_load_time = time.time() - model_load_start

    ndcg_scores = []
    mrr_scores = []
    queries = []

    encoding_time = 0
    scoring_time = 0

    grouped = df.groupby('query')

    for query, group in tqdm(grouped, desc="Evaluating queries"):
        queries.append(query)

        query_encode_start = time.time()
        query_vec = model.encode(query, convert_to_tensor=True, device=device)
        titles = group['title'].tolist()
        title_vecs = model.encode(titles, batch_size=batch_size, convert_to_tensor=True, device=device)
        encoding_time += time.time() - query_encode_start


        scoring_start = time.time()
        scores = util.cos_sim(query_vec, title_vecs)[0]
        relevances = torch.tensor(group['relevance'].to_numpy(), device=device)
        scores_np = scores.detach().cpu().numpy()
        relevances_np = relevances.detach().cpu().numpy()
        scoring_time += time.time() - scoring_start
        
        
        ndcg = compute_ndcg(relevances_np, scores_np, k=10)
        mrr = compute_mrr(relevances_np, scores_np, k=10)

        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)
    
    total_time = time.time() - start_time

    timing_info = {
        'total_execution_time': total_time,
        'model_load_time': model_load_time,
        'encoding_time': encoding_time,
        'scoring_time': scoring_time,
        'average_time_per_query': total_time / len(queries)
    }   

    # Build DataFrame
    results = pd.DataFrame({
        'query': queries,
        'NDCG@10': ndcg_scores,
        'MRR@10': mrr_scores
    })

    return results, timing_info