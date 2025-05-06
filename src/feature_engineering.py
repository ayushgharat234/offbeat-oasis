from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def prepare_location_features(locations_df):
    locations_df['combined_features'] = (
        locations_df['category'].fillna('') + ' ' +
        locations_df['state'].fillna('') + ' ' +
        locations_df['activities'].fillna('') + ' ' +
        locations_df['places'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(locations_df['combined_features'])
    
    # Changes Added ===========================================================================
    # Normalize numeric features
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(locations_df[['num_activities', 'num_places']])
    
    from scipy.sparse import hstack
    final_matrix = hstack([tfidf_matrix, numeric_features])

    # =========================================================================================

    return final_matrix, tfidf