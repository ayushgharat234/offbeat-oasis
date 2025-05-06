import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_user_location_matrix(reviews_df, users_df=None):
    interaction_matrix = reviews_df.pivot_table(
        index='user_id',
        columns='location_id',
        values='rating'
    ).fillna(0)
    return interaction_matrix


def get_top_k_similar_users(interaction_matrix, target_user_id, k=5):
    if target_user_id not in interaction_matrix.index:
        return None
    user_vector = interaction_matrix.loc[[target_user_id]]
    similarity_matrix = cosine_similarity(user_vector, interaction_matrix)[0]
    similarity_series = pd.Series(similarity_matrix, index=interaction_matrix.index)
    similarity_series = similarity_series.drop(target_user_id).sort_values(ascending=False).head(k)
    return similarity_series

def predict_ratings_for_user(interaction_matrix, similar_users, target_user_id):
    if similar_users is None:
        return pd.Series(dtype=float)
    neighbors_matrix = interaction_matrix.loc[similar_users.index]
    user_ratings = interaction_matrix.loc[target_user_id]
    weighted_ratings = neighbors_matrix.T.dot(similar_users)
    normalization = similar_users.sum()
    prediction_scores = weighted_ratings / normalization
    unseen_locations = user_ratings[user_ratings == 0].index
    return prediction_scores[unseen_locations].sort_values(ascending=False)