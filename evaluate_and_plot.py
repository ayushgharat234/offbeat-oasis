import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
from src.engine_wrapper import recommend_for_evaluation
from src.data_loader import load_all_data

locations_df, trips_df, users_df, reviews_df = load_all_data()

# Evaluation metrics
def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

def hit_rate_at_k(recommended, relevant, k):
    return int(any(item in relevant for item in recommended[:k]))

# Parameters
k_values = [3, 4, 5]
top_k_values = [5, 10, 15]
metrics = ['precision', 'recall', 'ndcg', 'hit_rate']
results = {}

# Run evaluation
for k_fold in k_values:
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    for top_k in top_k_values:
        scores = defaultdict(list)
        for train_idx, test_idx in kf.split(reviews_df):
            train = reviews_df.iloc[train_idx]
            test = reviews_df.iloc[test_idx]

            for user_id in test["user_id"].unique():
                user_train = train[train["user_id"] == user_id]
                user_test = test[test["user_id"] == user_id]

                if user_train.empty or user_test.empty:
                    continue

                relevant = user_test["location_id"].tolist()
                recs = recommend_for_evaluation(user_id, user_train, users_df, locations_df, trips_df, top_n=top_k)

                if recs is None or recs.empty:
                    continue

                predicted = recs["location_id"].tolist()
                scores["precision"].append(precision_at_k(predicted, relevant, top_k))
                scores["recall"].append(recall_at_k(predicted, relevant, top_k))
                scores["ndcg"].append(ndcg_at_k(predicted, relevant, top_k))
                scores["hit_rate"].append(hit_rate_at_k(predicted, relevant, top_k))

        results[(k_fold, top_k)] = {m: np.mean(scores[m]) for m in metrics}

# 🎨 Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for idx, metric in enumerate(metrics):
    ax = axs[idx]
    for k in k_values:
        y = [results[(k, tk)][metric] for tk in top_k_values]
        ax.plot(top_k_values, y, marker='o', label=f"K={k}")
    ax.set_title(f"{metric.upper()}@top_k")
    ax.set_xlabel("top_k")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("real_evaluation_metrics_plot.png")
plt.show()
