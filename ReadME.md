
This project benchmarks a pre-trained **SentenceTransformer** model for **semantic search retrieval** using two different datasets.

---

##  Datasets

- **Name**: Home Depot Product Search Relevance Dataset
- **Source**: [Kaggle Competition - Home Depot Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance)
- **Files Used**:
  - `train.csv` 
  - `product_descriptions.csv` 

Note: You must manually download these files from Kaggle and place them in your project folder.


- **Name: Home Depot Product Search Relevance Dataset
- **Source**: [BEIR Benchmark Download Page](https://github.com/beir-cellar/beir#download-datasets)
- **Files Used**: Files will be downloaded automatically.
  - `corpus.jsonl` 
  - `queries.jsonl` 
  - `qrels/test.tsv` 

Note: You do not need to download MS MARCO manually.
scripts/run_msmarco_eval.py will automatically fetch and unzip the dataset into the datasets/ folder on first run.


---

##  Model Used

- **Model Name**: `sentence-transformers/multi-qa-mpnet-base-cos-v1`
- **Model Type**: Pre-trained semantic search model from [Sentence-Transformers](https://www.sbert.net/)

---

##  Project Structure

```
semantic-search-evaluation/
├── src/
│   ├── helper.py        ← loading, corpus build, model, eval
│   ├── metrics.py       ← NDCG, MRR, Recall implementations
│   └── config.py        ← central constants
├── scripts/
│   ├── run_home_depot_eval.py   ← evaluate Home‑Depot
│   └── run_msmarco_eval.py      ← evaluate MS MARCO
├── outputs/              ← CSV metrics
├── datasets/             ← raw data (git‑ignored)
├── requirements.txt
└── README.md
```
---

## ⚡ Quick Start: How to Reproduce the Results

1. **Clone this repository**:
    ```bash
    git clone https://github.com/Mbalavar/semantic-search-evaluation.git
    cd semantic-search-evaluation
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
### Evaluation Stage 1: Home Depot Dataset

1. Download the Home Depot Dataset from [Kaggle](https://www.kaggle.com/c/home-depot-product-search-relevance/data) and place:
    - `train.csv`
    - `product_descriptions.csv`
    into the project root or `datasets/` directory.

2. Update file paths if necessary inside `run_home_depot_eval.py`.

3. Run the evaluation pipeline:
    ```bash
    python scripts/run_home_depot_eval.py
    ```

This will:
- Preprocess and clean the data.
- Filter queries with ≥10 candidate products.
- Create test samples (stratified 50% sample of the original queries + 20000 distractors).
- Encode and evaluate queries using the pre-trained model.
- Save the evaluation scores to `outputs/home_depot_eval.csv`.

---

### Evaluation Stage 2: MS MARCO Dataset

1. Run the evaluation script:
    ```bash
    python scripts/run_msmarco_eval.py
    ```

This will:
- Download and unzip the BEIR-formatted MS MARCO dataset.
- Sample 50% of queries and retain all relevant passages + 100000 distractors.
- Encode and rank passages using the pre-trained model.
- Compute evaluation metrics (NDCG@10, MRR@10, Recall@10).
- Save results to `outputs/msmarco_eval.csv`.

To change sample ratio, number of distractors, or model, update `src/config.py`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@10** | Normalized Discounted Cumulative Gain at 10 (ranking quality) |
| **MRR@10**  | Mean Reciprocal Rank at 10 (first relevant result rank) |
| **Recall@10** | Recall at 10 (fraction of relevant queries with at least one relevant result) |

---

## Notes

- Home Depot evaluation uses only product titles (not descriptions).
- Sampling ensures fast evaluation; adjust parameters in `config.py`.
- GPU is used automatically if available.

---

## Author

- **Name**: Mohsen Balavar
