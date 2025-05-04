"""
Run evaluation on the Home‑Depot Kaggle dataset
(assumes raw CSVs are in project root)
"""


import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]   # one level up from scripts/
sys.path.append(str(repo_root))  


import os
from src.config  import (MODEL_NAME, BATCH_SIZE, K_VALUES,
                         HOME_QUERY_RATIO, HOME_DISTRACTORS, SEED)
from src.helper  import (load_and_preprocess_home, load_model,
                         eval_dataset)

RAW_PRODUCTS   = "datasets/train.csv"
RAW_DESCS      = "datasets/product_descriptions.csv"
OUT_METRICS_CSV = "outputs/home_depot_eval.csv"

# ------------------------------------------------------------------
# 1. prepare cleaned test set
df = load_and_preprocess_home(RAW_PRODUCTS, RAW_DESCS,
                              out_csv="datasets/home_depot_clean.csv")

# Convert DF to dict format expected by eval_dataset
corpus = {}
for i, row in df.iterrows():
    corpus[str(i)] = {     
        "text":  row["title"]           # only title used
    }

queries = {row["query"]: row["query"] for _, row in df.iterrows()}

# Build qrels in dict→dict form
qrels = {}
for idx, row in df.iterrows():
    qrels.setdefault(row["query"], {})[str(idx)] = int(row["relevance"])

# ------------------------------------------------------------------
# 2. Model + evaluation
model = load_model()
ndcg, mrr, recall = eval_dataset(model, corpus, queries, qrels)
# ------------------------------------------------------------------

# 3. Save CSV & print
os.makedirs("outputs", exist_ok=True)
with open(OUT_METRICS_CSV, "w") as f:
    f.write("Dataset,NDCG@10,MRR@10,Recall@10\n")
    f.write(f"home_depot,{ndcg:.4f},{mrr:.4f},{recall:.4f}\n")

print("\n===== Home‑Depot =====")
print(f"NDCG@10 : {ndcg:.4f}")
print(f"MRR@10  : {mrr:.4f}")
print(f"Recall@10: {recall:.4f}")
print(f"CSV saved → {OUT_METRICS_CSV}")