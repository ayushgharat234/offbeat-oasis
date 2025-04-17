from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_user_location_matrix(reviews_df):
    return reviews_df.pivot_table(index="user_id", columns="location_id", values="rating").fillna(0)

def get_top_k_similar_users(interaction_matrix, user_id, k=5):
    user_vector = interaction_matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, interaction_matrix.values)[0]
    similar_indices = similarities.argsort()[::-1][1:k+1]
    return interaction_matrix.index[similar_indices]

def predict_ratings_for_user(interaction_matrix, similar_users, target_user_id):
    return interaction_matrix.loc[similar_users].mean()