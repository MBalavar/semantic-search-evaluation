
This project benchmarks a pre-trained **SentenceTransformer** model for **semantic search retrieval** using the Home Depot product dataset.

---

## 📚 Dataset

- **Name: Home Depot Product Search Relevance Dataset
- **Source**: [Kaggle Competition - Home Depot Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance)
- **Files Used**:
  - `train.csv` 
  - `product_descriptions.csv` 

Note: You must manually download these files from Kaggle and place them in your project folder.

---

## 🤖 Model Used

- **Model Name**: `sentence-transformers/multi-qa-mpnet-base-cos-v1`
- **Model Type**: Pre-trained semantic search model from [Sentence-Transformers](https://www.sbert.net/)

---

## 📂 Project Structure


semantic-search-evaluation-home-depot/
├── src/
│   ├── init.py
│   ├── helpers.py        # Preprocessing, merging, cleaning
│   ├── metrics.py        # NDCG@10, MRR@10 computations
│   └── evaluate.py       # Evaluation logic (batch, GPU support)
├── results.csv           # Evaluation results
├── run_eval.py           # Launcher script to run the full evaluation
├── requirements.txt      # Required Python libraries
├── README.md
├── .gitignore


---

## ⚡ Quick Start: How to Reproduce the Results

1. **Clone this repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/semantic-search-evaluation-home-depot.git
    cd semantic-search-evaluation-home-depot
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

4. **Download the Home Depot Dataset** from [Kaggle](https://www.kaggle.com/c/home-depot-product-search-relevance/data) and place:
    - `train.csv`
    - `product_descriptions.csv`
    into the project root.

5. **Update file paths if necessary** inside `run_eval.py`.

6. **Run the evaluation pipeline**:
    ```bash
    python run_eval.py
    ```

✅ This will:
- Preprocess and clean the data.
- Merge product titles and descriptions.
- Create test samples (stratified).
- Encode and evaluate queries using the pre-trained model.
- Save the evaluation scores to `/results/result.csv`.

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@10** | Normalized Discounted Cumulative Gain at 10 (ranking quality) |
| **MRR@10**  | Mean Reciprocal Rank at 10 (first relevant result rank) |

---

## 📢 Notes

- Evaluation is performed on a stratified 20% sample of the original queries.
- Only queries with **≥10** candidate products are included to ensure meaningful evaluation.

---

## 👨‍💻 Author

- **Name**: [Mohsen Balavar]
