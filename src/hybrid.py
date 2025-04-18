from .utils import normalize_scores

def combine_scores(content_df, collab_scores, weight_content=0.5, weight_collab=0.5):
    content_df = content_df.copy()
    content_df["normalized_content"] = normalize_scores(content_df["content_score"])

    collab_df = collab_scores.reset_index()
    collab_df.columns = ["location_id", "collab_score"]
    collab_df["normalized_collab"] = normalize_scores(collab_df["collab_score"])

    merged = content_df.merge(collab_df, on="location_id", how="left").fillna(0)
    merged["hybrid_score"] = (
        weight_content * merged["normalized_content"] + weight_collab * merged["normalized_collab"]
    )
    return merged.sort_values("hybrid_score", ascending=False)