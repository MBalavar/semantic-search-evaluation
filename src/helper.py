# src/helpers.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Simple text cleaning: lowercase, remove HTML tags, punctuation, and extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)       # remove extra spaces
    return text.strip()

def map_relevance(score):
    """Map continuous relevance to 0, 1, 2, 3."""
    if score < 1.5:
        return 0
    elif score < 2.0:
        return 1
    elif score < 2.5:
        return 2
    else:
        return 3

def load_and_preprocess(products_path, descriptions_path):
    """
    Load and preprocess Home Depot train.csv, product_descriptions.csv.
    Returns a clean dataframe ready for evaluation.
    """

    # Load datasets
    df_products = pd.read_csv(products_path, encoding='latin-1')
    df_desc = pd.read_csv(descriptions_path)


    # Merge product description
    df = pd.merge(df_products, df_desc[['product_uid', 'product_description']], on='product_uid', how='left')

    # Text cleaning
    df['search_term'] = df['search_term'].apply(clean_text)
    df['product_title'] = df['product_title'].apply(clean_text)
    df['product_description'] = df['product_description'].apply(clean_text)

    # Map relevance scores to classes
    df['relevance_mapped'] = df['relevance'].apply(map_relevance)

    # Filter queries with enough results (>=10 results)
    query_counts = df.groupby('search_term').size().reset_index(name='num_results')
    valid_queries = query_counts[query_counts['num_results'] >= 10]['search_term']
    df = df[df['search_term'].isin(valid_queries)]


    # Split into stratified train/test
    train_stratified, test_stratified = train_test_split(
        df, test_size=0.2, stratify=df['relevance_mapped'], random_state=42
    )

    # Keep only relevant columns for test set
    df = test_stratified[['search_term', 'product_title', 'product_description', 'relevance']]
    df.columns = ['query', 'title', 'description', 'relevance']

    df.to_csv('./test_clean.csv', index=False)
    print("Saved cleaned test set to 'test_clean.csv'.")

    return df