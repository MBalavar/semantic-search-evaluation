import os
import sys



from src.helper import load_and_preprocess
from src.evaluate import evaluate

# Paths to your raw data
products_path = './train.csv'
descriptions_path = './product_descriptions.csv'

# Load, preprocess, and save cleaned test set
df_test = load_and_preprocess(products_path, descriptions_path)

# Evaluate using the cleaned test set
results = evaluate(df_test)


# Save evaluation results
results.to_csv('results/result.csv', index=False)

# Calculate and print average metrics
avg_ndcg = results['NDCG@10'].mean()
avg_mrr = results['MRR@10'].mean()

print("\n===== Evaluation Results =====")
print(f"Average NDCG@10: {avg_ndcg:.4f}")
print(f"Average MRR@10: {avg_mrr:.4f}")
print("==============================\n")

print("✅ Evaluation completed and results saved.")