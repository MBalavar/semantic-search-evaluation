"""
Run evaluation on the BEIR‑formatted MS MARCO passage dataset.
"""

# Add project root to Python path so src modules can be imported
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

# Standard imports
import os

# Import configuration constants
from src.config import (MODEL_NAME, BATCH_SIZE, K_VALUES,
                        MSMARCO_QUERY_RATIO, MSMARCO_DISTRACTORS, SEED)

# Import helper functions from src
from src.helper import (download_dataset, load_sample,
                        build_corpus, load_model, eval_dataset)

# Dataset and output file names
DATASET_NAME     = "msmarco"
OUT_METRICS_CSV  = "outputs/msmarco_eval.csv"

# ------------------------------------------------------------------
# Step 1: Download MS MARCO dataset (BEIR format)
download_dataset(DATASET_NAME)

# Step 2: Load a random sample of queries + their qrels 
corpus, queries, qrels = load_sample(
        f"datasets/{DATASET_NAME}",
        ratio=MSMARCO_QUERY_RATIO,
        seed=SEED)

# Step 3: Build a filtered corpus: include all relevant docs + N distractors
corpus = build_corpus(corpus, qrels, MSMARCO_DISTRACTORS, SEED)

# ------------------------------------------------------------------
# Step 4: Load the SentenceTransformer model
model  = load_model()

# Step 5: Run retrieval and compute metrics
ndcg, mrr, recall = eval_dataset(model, corpus, queries, qrels)

# ------------------------------------------------------------------
# Step 6: Save results to CSV
os.makedirs("outputs", exist_ok=True)
with open(OUT_METRICS_CSV, "w") as f:
    f.write("Dataset,NDCG@10,MRR@10,Recall@10\n")
    f.write(f"msmarco,{ndcg:.4f},{mrr:.4f},{recall:.4f}\n")

# Step 7: Print metrics
print("\n===== MS‑MARCO =====")
print(f"NDCG@10 : {ndcg:.4f}")
print(f"MRR@10  : {mrr:.4f}")
print(f"Recall@10: {recall:.4f}")
print(f"CSV saved → {OUT_METRICS_CSV}")