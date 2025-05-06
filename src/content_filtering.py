from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import numpy as np

"""
def create_user_preference_vector(travel_category, preferred_state, tfidf):
    user_text = f"{travel_category} {preferred_state}"
    user_vector = tfidf.transform([user_text])
    return user_vector
"""

def create_user_preference_vector(travel_category, preferred_state, tfidf, user_metadata):
    # Create text vector from user metadata
    user_text = f"{travel_category} {preferred_state} {user_metadata['occupation']} {user_metadata['location_type']}"
    tfidf_vector = tfidf.transform([user_text])  # shape: (1, N)

    # Normalized numeric dummy features (as sparse row vector)
    numeric_vector = csr_matrix([[1.0, 1.0]])  # shape: (1, 2)

    # Combine text + numeric into final vector
    final_user_vector = hstack([tfidf_vector, numeric_vector])
    return final_user_vector


def get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=10):
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    locations_df = locations_df.copy()
    locations_df['content_score'] = similarity_scores
    return locations_df.sort_values(by='content_score', ascending=False).head(top_n)