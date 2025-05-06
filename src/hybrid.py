from src.utils import normalize_scores

def combine_scores(content_df, collab_scores, user_id, reviews_df, weight_content=0.5, weight_collab=0.5):
    # Adjust weights dynamically
    num_reviews = len(reviews_df[reviews_df['user_id'] == user_id])
    if num_reviews < 3:
        weight_content, weight_collab = 0.8, 0.2
    elif num_reviews > 10:
        weight_content, weight_collab = 0.3, 0.7

    content_df = content_df.copy()
    content_df["normalized_content"] = normalize_scores(content_df["content_score"])

    collab_df = collab_scores.reset_index()
    collab_df.columns = ["location_id", "collab_score"]
    collab_df["normalized_collab"] = normalize_scores(collab_df["collab_score"])

    # 🛠️ Force consistent types
    content_df["location_id"] = content_df["location_id"].astype(int)
    collab_df["location_id"] = collab_df["location_id"].astype(int)

    merged = content_df.merge(collab_df, on="location_id", how="left").fillna(0)
    merged["hybrid_score"] = (
        weight_content * merged["normalized_content"] + weight_collab * merged["normalized_collab"]
    )

    return merged.sort_values("hybrid_score", ascending=False)
