from src.data_loader import load_all_data
from src.feature_engineering import prepare_location_features
from src.content_filtering import create_user_preference_vector, get_content_based_recommendations
from src.collaborative_filtering import create_user_location_matrix, get_top_k_similar_users, predict_ratings_for_user
from src.utils import estimate_location_cost, apply_budget_filter
from src.hybrid import combine_scores

if __name__ == "__main__":
    travel_category = "Adventure"
    preferred_state = "Himachal Pradesh"
    trip_budget = 40000
    target_user_id = 30
    top_k = 10

    locations_df, trips_df, users_df, reviews_df = load_all_data()
    tfidf_matrix, tfidf = prepare_location_features(locations_df)
    user_vector = create_user_preference_vector(travel_category, preferred_state, tfidf)
    content_recs = get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=50)

    interaction_matrix = create_user_location_matrix(reviews_df)
    similar_users = get_top_k_similar_users(interaction_matrix, target_user_id, k=5)
    collab_scores = predict_ratings_for_user(interaction_matrix, similar_users, target_user_id)

    location_costs = estimate_location_cost(reviews_df, trips_df)
    filtered = apply_budget_filter(content_recs, location_costs, trip_budget)
    final_recommendations = combine_scores(filtered, collab_scores, weight_content=0.6, weight_collab=0.4)

    print(final_recommendations[["location_id", "hybrid_score"]].head(top_k))