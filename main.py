import src.data_loader as dl
import src.feature_engineering as fe
import src.content_filtering as cb
import src.collaborative_filtering as cf
import src.utils as ut
import src.hybrid as hy

if __name__ == "__main__":
    travel_category = "Adventure"
    preferred_state = "Himachal Pradesh"
    trip_budget = 40000
    target_user_id = 30
    top_k = 10

    # Load data
    locations_df, trips_df, users_df, reviews_df = dl.load_all_data()

    # Feature engineering
    tfidf_matrix, tfidf = fe.prepare_location_features(locations_df)
    user_vector = cb.create_user_preference_vector(travel_category, preferred_state, tfidf)
    content_recs = cb.get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=50)

    # Collaborative filtering
    interaction_matrix = cf.create_user_location_matrix(reviews_df)
    similar_users = cf.get_top_k_similar_users(interaction_matrix, target_user_id, k=5)
    collab_scores = cf.predict_ratings_for_user(interaction_matrix, similar_users, target_user_id)

    # Budget filtering
    location_costs = ut.estimate_location_cost(reviews_df, trips_df)
    filtered = ut.apply_budget_filter(content_recs, location_costs, trip_budget)

    # Combine scores
    final_recommendations = hy.combine_scores(filtered, collab_scores, weight_content=0.6, weight_collab=0.4)

    # Show output
    print(final_recommendations[["location_id", "hybrid_score"]].head(top_k))