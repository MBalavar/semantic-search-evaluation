# src/helper.py
# -------------------------------------------------
#
# Unified helper module for:
#   • Home‑Depot (CSV‑based)
#   • BEIR/MS MARCO (BEIR format)
#
# -------------------------------------------------

import os, re, csv, time, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import util as st_util

from src.metrics import ndcg, mrr, recall_at_k
from src.config  import (MODEL_NAME, BATCH_SIZE, DEVICE, K_VALUES, SEED)

# =================================================
# ------- 1.  Home‑Depot CSV utilities ------------
# =================================================

def clean_text(text: str) -> str:
    """Lower‑case, strip HTML, punctuation, extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)        # HTML tags
    text = re.sub(r"[^\w\s]", " ", text)        # punctuation
    text = re.sub(r"\s+", " ", text)            # multiple spaces
    return text.strip()

def map_relevance(score: float) -> int:
    """Map 1-3 float relevance to 0-3 integer bins."""
    if score < 1.5:
        return 0
    if score < 2.0:
        return 1
    if score < 2.5:
        return 2
    return 3

def load_and_preprocess_home(products_csv: str,
                             descriptions_csv: str,
                             out_csv: str = "outputs/home_test_clean.csv",
                             min_results: int = 10,
                             test_ratio: float = 0.2,
                             seed: int = SEED):
    """Return cleaned test-set DataFrame for Home-Depot evaluation."""
    df_prod = pd.read_csv(products_csv, encoding="latin-1")
    df_desc = pd.read_csv(descriptions_csv)

    df = pd.merge(df_prod,
                  df_desc[["product_uid", "product_description"]],
                  on="product_uid",
                  how="left")

    # Basic cleaning
    for col in ("search_term", "product_title", "product_description"):
        df[col] = df[col].apply(clean_text)

    df["relevance_cls"] = df["relevance"].apply(map_relevance)

    # Keep only queries with ≥min_results documents
    counts = df.groupby("search_term").size().reset_index(name="num")
    valid  = counts[counts["num"] >= min_results]["search_term"]
    df     = df[df["search_term"].isin(valid)]

    train, test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["relevance_cls"],
        random_state=seed,
    )

    test = test[["search_term", "product_title",
                 "product_description", "relevance"]]
    test.columns = ["query", "title", "description", "relevance"]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    test.to_csv(out_csv, index=False)
    print(f"Saved cleaned Home-Depot test set → {out_csv}")
    return test

# =================================================
# ------- 2.  Generic BEIR utilities --------------
# =================================================

def download_dataset(dataset_name: str):
    """Download & unzip a BEIR dataset if not already present."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    util.download_and_unzip(url, "datasets")

def load_sample(dataset_path: str,
                ratio: float,
                seed: int = SEED):
    """Load BEIR split, sample queries according to ratio."""
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    random.seed(seed)
    sample_n = max(1, int(ratio * len(queries)))
    q_ids    = random.sample(list(queries.keys()), sample_n)
    queries  = {q: queries[q] for q in q_ids}
    qrels    = {q: qrels[q] for q in q_ids if q in qrels}
    return corpus, queries, qrels

def build_corpus(corpus: dict,
                 qrels: dict,
                 distractor_size: int,
                 seed: int = SEED):
    """Keep all relevant docs + random distractors."""
    rel_ids = {doc for q in qrels.values() for doc in q.keys()}
    remaining = list(set(corpus.keys()) - rel_ids)
    random.seed(seed)
    distractors = random.sample(remaining,
                                k=min(distractor_size, len(remaining)))
    final_ids = rel_ids.union(distractors)
    return {d: corpus[d] for d in final_ids}

# =================================================
# ------- 3.  Model & evaluation helpers ----------
# =================================================

def load_model():
    model = DRES(models.SentenceBERT(MODEL_NAME), batch_size=BATCH_SIZE)
    if torch.cuda.is_available() and DEVICE == "cuda":
        model.model = model.model.to("cuda")
        print("Using GPU")
    else:
        print("Running on CPU")
    return model

def eval_dataset(model,
                 corpus: dict,
                 queries: dict,
                 qrels: dict,
                 k: int = K_VALUES[0]):
    """Return (NDCG, MRR, Recall) averages for top-k."""
    t0 = time.time()
    results = model.search(corpus, queries,
                           top_k=k, score_function="cos_sim")
    print(f"Retrieval took {time.time()-t0:.2f}s")

    ndcg_list, mrr_list, rec_list = [], [], []

    for qid, doc_scores in results.items():
        scores = np.array(list(doc_scores.values()))
        rels   = np.array([qrels[qid].get(did, 0) for did in doc_scores.keys()])
        ndcg_list.append(ndcg(rels, scores, k))
        mrr_list.append(mrr(rels, scores, k))
        rec_list.append(recall_at_k(rels, scores, k))

    return (float(np.mean(ndcg_list)),
            float(np.mean(mrr_list)),
            float(np.mean(rec_list)))