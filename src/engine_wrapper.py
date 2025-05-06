from src.feature_engineering import prepare_location_features
from src.content_filtering import create_user_preference_vector, get_content_based_recommendations
from src.collaborative_filtering import create_user_location_matrix, get_top_k_similar_users, predict_ratings_for_user
from src.utils import estimate_location_cost, get_user_budget_limit
from src.hybrid import combine_scores
from src import config

def recommend_for_evaluation(user_id, reviews_df, users_df, locations_df, trips_df, top_n=10):
    try:
        user_metadata = users_df[users_df["user_id"] == user_id].iloc[0]
    except IndexError:
        return None

    # Feature engineering
    tfidf_matrix, tfidf = prepare_location_features(locations_df)
    user_vector = create_user_preference_vector("Adventure", "Goa", tfidf, {
        "occupation": user_metadata["occupation"],
        "location_type": user_metadata["location_type"]
    })

    content_recs = get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=50)

    # Collaborative filtering
    interaction_matrix = create_user_location_matrix(reviews_df, users_df)
    similar_users = get_top_k_similar_users(interaction_matrix, user_id, k=5)
    collab_scores = predict_ratings_for_user(interaction_matrix, similar_users, user_id)

    # Budget filtering
    location_costs = estimate_location_cost(reviews_df, trips_df)
    user_budget = get_user_budget_limit(users_df, user_id)
    content_recs = content_recs.merge(location_costs, on="location_id", how="left")
    filtered_recs = content_recs[content_recs["estimated_cost"] <= user_budget]

    final_recs = combine_scores(
        filtered_recs,
        collab_scores,
        user_id,
        reviews_df,
        weight_content=config.WEIGHT_CONTENT,
        weight_collab=config.WEIGHT_COLLAB
    )

    return final_recs[["location_id", "hybrid_score"]].head(top_n)