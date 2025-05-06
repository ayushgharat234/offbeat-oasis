
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from src.data_loader import load_all_data
from src.engine_wrapper import recommend_for_evaluation

# Evaluation metrics
def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg != 0 else 0.0

def hit_rate_at_k(recommended, relevant, k):
    return int(any(item in relevant for item in recommended[:k]))

# Main evaluation with K-Fold CV
def run_kfold_evaluation(k=5, top_k=10):
    locations_df, trips_df, users_df, reviews_df = load_all_data()

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = defaultdict(list)

    fold = 1
    for train_idx, test_idx in kf.split(reviews_df):
        print(f"Evaluating Fold {fold}...")
        train_reviews = reviews_df.iloc[train_idx]
        test_reviews = reviews_df.iloc[test_idx]
        test_users = test_reviews["user_id"].unique()

        for user_id in test_users:
            user_test = test_reviews[test_reviews["user_id"] == user_id]
            user_train = train_reviews[train_reviews["user_id"] == user_id]

            if user_test.empty or user_train.empty:
                continue

            relevant_locs = user_test["location_id"].tolist()
            recs = recommend_for_evaluation(user_id, user_train, users_df, locations_df, trips_df, top_n=top_k)

            if recs is None or recs.empty:
                continue

            predicted_locs = recs["location_id"].tolist()

            metrics["precision"].append(precision_at_k(predicted_locs, relevant_locs, top_k))
            metrics["recall"].append(recall_at_k(predicted_locs, relevant_locs, top_k))
            metrics["ndcg"].append(ndcg_at_k(predicted_locs, relevant_locs, top_k))
            metrics["hit_rate"].append(hit_rate_at_k(predicted_locs, relevant_locs, top_k))

        fold += 1

    print("\n=== Final K-Fold Evaluation Results ===")
    for metric, scores in metrics.items():
        print(f"{metric}@{top_k}: {np.mean(scores):.4f}")

if __name__ == "__main__":
    run_kfold_evaluation(k=3, top_k=5)
