import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load your review data
reviews = pd.read_csv("data/final/reviews.csv")

# Create interaction matrix (user x location with ratings)
interaction_matrix = reviews.pivot_table(index='user_id', columns='location_id', values='rating').fillna(0)

# Split reviews into train/test per user
def split_train_test(reviews, test_size=0.25):
    train_rows, test_rows = [], []
    for user_id, user_data in reviews.groupby('user_id'):
        if len(user_data) >= 2:
            test_sample = user_data.sample(frac=test_size, random_state=42)
            train_sample = user_data.drop(test_sample.index)
            train_rows.append(train_sample)
            test_rows.append(test_sample)
        else:
            train_rows.append(user_data)
    return pd.concat(train_rows), pd.concat(test_rows)

train_reviews, test_reviews = split_train_test(reviews)

# Create train interaction matrix
train_matrix = train_reviews.pivot_table(index='user_id', columns='location_id', values='rating').fillna(0)

# Get top K similar users
def get_top_k_similar_users(matrix, user_id, k=5):
    if user_id not in matrix.index:
        return []
    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(matrix.values, user_vector).flatten()
    indices = similarities.argsort()[::-1]
    similar_users = [matrix.index[i] for i in indices if matrix.index[i] != user_id][:k]
    return similar_users

# Predict scores from similar users
def predict_scores_for_user(matrix, similar_users):
    return matrix.loc[similar_users].mean()

# Evaluation metrics
def precision_at_k(preds, actuals, k):
    hits = len(set(preds[:k]) & set(actuals))
    return hits / k

def recall_at_k(preds, actuals, k):
    hits = len(set(preds[:k]) & set(actuals))
    return hits / len(actuals) if actuals else 0

# Evaluate across users
def evaluate(train_matrix, test_reviews, k=5):
    precision_list, recall_list = [], []
    for user_id in test_reviews['user_id'].unique():
        if user_id not in train_matrix.index:
            continue

        actual_items = test_reviews[test_reviews['user_id'] == user_id]['location_id'].tolist()
        similar_users = get_top_k_similar_users(train_matrix, user_id, k=5)
        if not similar_users:
            continue

        predicted_scores = predict_scores_for_user(train_matrix, similar_users)
        recommended_items = predicted_scores.sort_values(ascending=False).index.tolist()

        precision = precision_at_k(recommended_items, actual_items, k)
        recall = recall_at_k(recommended_items, actual_items, k)

        precision_list.append(precision)
        recall_list.append(recall)

    return {
        "precision@{}".format(k): sum(precision_list) / len(precision_list),
        "recall@{}".format(k): sum(recall_list) / len(recall_list)
    }

# Run evaluation
metrics = evaluate(train_matrix, test_reviews, k=5)
print("\n📊 Evaluation Results:")
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")