from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_location_features(locations_df):
    locations_df['combined_features'] = (
        locations_df['category'].fillna('') + ' ' +
        locations_df['state'].fillna('') + ' ' +
        locations_df['activities'].fillna('') + ' ' +
        locations_df['places'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(locations_df['combined_features'])
    return tfidf_matrix, tfidf