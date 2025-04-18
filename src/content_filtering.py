from sklearn.metrics.pairwise import cosine_similarity

def create_user_preference_vector(travel_category, preferred_state, tfidf):
    user_text = f"{travel_category} {preferred_state}"
    user_vector = tfidf.transform([user_text])
    return user_vector

def get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=10):
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    locations_df = locations_df.copy()
    locations_df['content_score'] = similarity_scores
    return locations_df.sort_values(by='content_score', ascending=False).head(top_n)