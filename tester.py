import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# New list of actual offbeat locations
actual_locations_data = {
    'location_name': [
        'Chitkul', 'Landour', 'Gurez Valley', 'Shojha', 'Pangong Tso West Bank Villages',
        'Kalpa', 'Munsiyari', 'Tirthan Valley', 'Naukuchiatal', 'Shekhawati Region',
        'Chopta-Tungnath-Chandrashila', 'Losar', 'Gandikota', 'Dhanushkodi', 'Athirappilly And Vazhachal Waterfalls',
        "Hampi'S Lesser-Known Ruins", "Coorg'S Offbeat Trails And Villages", 'Poovar Island', 'Valparai', 'Araku Valley',
        'Thenmala', 'Lepakshi', 'Kudremukh', 'Meghamalai', 'Bhandardara',
        'Diu Island (Quieter Beaches)', "Mount Abu'S Offbeat Trails", 'Kachchh', 'Malshej Ghat', 'Saputara',
        'Lonar Crater Lake', 'Jawhar', 'Champaner-Pavagadh Archaeological Park', 'Tarkarli', 'Polo Forest',
        'Amboli', 'Pachmarhi', 'Orchha', 'Khajuraho', 'Bhedaghat (Beyond The Marble Rocks)',
        'Mandu', "Kanha National Park'S Buffer Zones", 'Ziro Valley', 'Mawlynnong', 'Champhai',
        'Tawang', 'Jaldapara National Park', 'Daringbadi', 'Pelling', 'Unakoti',
        'Majuli Island', 'Sandakphu-Phalut Trek (Less Crowded Sections)', 'Mokokchung', 'Haflong', 'Gaya',
        'Sirpur', 'Hissar', 'Ganjam District', 'Rurnagar', 'Giridih District',
        'Valmiki National Park', 'Agonda', 'Karwar', 'Rohtak', 'Ludhiana',
        'Sinquerim', 'Surguja', 'Dhanbad', 'Udupi', 'Rajgir',
        'Varkala', 'Havelock Island', 'Faridabad', 'Long Island', 'Ukhrul',
        'Balasore', 'Bastar (District)', 'Cuttack', 'Canacona', 'Kalimpong',
        'Bhagalpur', 'Bhilai', 'Tirthan Valley', 'Silvassa', 'Kavaratti Island',
        'Yanam', 'Ranchi', 'Pondicherry', 'Lucknow', 'Imphal West',
        'Secunderabad', 'Brahmapur', 'Panchkula', 'Chitrakoot District', 'Warangal', 'Dimapur',
        'Latehar District', 'Mohali', 'Sambalpur', 'Kohima', 'Maheshwar',
        'Osian', 'Kannur', 'Kawardha', 'Gandikota,', 'Konkan'
    ],
    'state': [
        'Himachal Pradesh', 'Uttarakhand', 'Jammu And Kashmir', 'Himachal Pradesh', 'Ladakh',
        'Himachal Pradesh', 'Uttarakhand', 'Himachal Pradesh', 'Uttarakhand', 'Rajasthan',
        'Uttarakhand', 'Himachal Pradesh', 'Andhra Pradesh', 'Tamil Nadu', 'Kerala',
        'Karnataka', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Andhra Pradesh',
        'Kerala', 'Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Maharashtra',
        'Gujarat', 'Rajasthan', 'Gujarat', 'Maharashtra', 'Gujarat',
        'Maharashtra', 'Maharashtra', 'Gujarat', 'Maharashtra', 'Gujarat',
        'Maharashtra', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh',
        'Madhya Pradesh', 'Madhya Pradesh', 'Arunachal Pradesh', 'Meghalaya', 'Mizoram',
        'Arunachal Pradesh', 'West Bengal', 'Odisha', 'Sikkim', 'Tripura',
        'Assam', 'West Bengal', 'Nagaland', 'Assam', 'Bihar',
        'Chhattisgarh', 'Haryana', 'Odisha', 'Punjab', 'Jharkhand',
        'Bihar', 'Goa', 'Karnataka', 'Haryana', 'Punjab',
        'Goa', 'Chhattisgarh', 'Jharkhand', 'Karnataka', 'Bihar',
        'Kerala', 'Andaman and Nicobar Islands', 'Haryana', 'Andaman and Nicobar Islands', 'Manipur',
        'Odisha', 'Chhattisgarh', 'Odisha', 'Goa', 'West Bengal',
        'Bihar', 'Chhattisgarh', 'Himachal Pradesh', 'Dadra and Nagar Haveli', 'Lakshadweep',
        'Puducherry (Union Territory)', 'Jharkhand', 'Puducherry (Union Territory)', 'Uttar Pradesh', 'Manipur',
        'Telangana', 'Odisha', 'Haryana', 'Uttar Pradesh', 'Telangana', 'Nagaland',
        'Jharkhand', 'Punjab', 'Odisha', 'Nagaland', 'Madhya Pradesh',
        'Rajasthan', 'Kerala', 'Chhattisgarh', 'Andra Pradesh', 'Maharashtra'
    ]
}
locations_df = pd.DataFrame(actual_locations_data)
locations_df['location_id'] = [f'L{i+1}' for i in range(len(locations_df))]
locations_df = locations_df[['location_id', 'location_name', 'state']] # Reorder columns

# --- Generate synthetic users (adjusting preferred locations) ---
users_data = []
num_users = 150 # Increased number of users
for i in range(num_users):
    preferred_state = np.random.choice(locations_df['state'].unique())
    state_locations = locations_df[locations_df['state'] == preferred_state]['location_name']
    num_possible_locations = len(state_locations)
    num_preferred = np.random.randint(1, min(4, num_possible_locations + 1)) # Ensure we don't ask for more than exist
    preferred_locations = np.random.choice(state_locations, size=num_preferred, replace=False).tolist()
    users_data.append({'user_id': f'U{i+1}', 'preferred_state': preferred_state, 'preferred_locations': preferred_locations})
users_df = pd.DataFrame(users_data)

# --- Generate synthetic user-location interactions ---
interactions_data = []
num_interactions = 700 # Increased number of interactions
for _ in range(num_interactions):
    user = users_df.sample(n=1).iloc[0]
    possible_locations = locations_df[locations_df['state'] == user['preferred_state']]['location_name'].tolist()
    if possible_locations:
        location_name = np.random.choice(possible_locations)
        location_id = locations_df[locations_df['location_name'] == location_name]['location_id'].iloc[0]
        visit_count = np.random.randint(1, 6)
        interactions_data.append({'user_id': user['user_id'], 'location_id': location_id, 'visit_count': visit_count})
interactions_df = pd.DataFrame(interactions_data)

# --- Add dummy descriptions for content-based filtering (replace with actual data later) ---
locations_df['description'] = [f"An offbeat location in {row['state']}" for index, row in locations_df.iterrows()]

# --- 1. Content-Based Filtering ---
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(locations_df['description'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
location_tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=locations_df['location_id'], columns=tfidf_feature_names)

def get_content_based_recommendations(user_id, users_df, locations_df, location_tfidf_df, top_n=5):
    user = users_df[users_df['user_id'] == user_id].iloc[0]
    preferred_state = user['preferred_state']

    # For simplicity, we'll recommend based on state for now, as we don't have place types
    eligible_locations = locations_df[locations_df['state'] == preferred_state]
    if eligible_locations.empty:
        return []

    # Calculate similarity based on dummy descriptions
    user_profile = f"offbeat location in {preferred_state}"
    user_profile_vector = tfidf_vectorizer.transform([user_profile]).toarray()
    similarity_scores = cosine_similarity(user_profile_vector, location_tfidf_df)[0]
    similarity_df = pd.DataFrame({'location_id': location_tfidf_df.index, 'similarity': similarity_scores})
    merged_df = pd.merge(similarity_df, eligible_locations, on='location_id')
    ranked_recommendations = merged_df.sort_values(by='similarity', ascending=False).head(top_n)['location_id'].tolist()
    return ranked_recommendations

# --- 2. User-Based Collaborative Filtering ---
user_item_matrix = interactions_df.pivot_table(index='user_id', columns='location_id', values='visit_count').fillna(0)
user_similarity_matrix = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, top_n=5):
    if user_id not in user_similarity_df.index or user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).drop(user_id)
    user_visited_locations = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    recommendations = defaultdict(float)

    for other_user, similarity in similar_users.head(10).items():
        visited_by_other = user_item_matrix.loc[other_user][user_item_matrix.loc[other_user] > 0].index.tolist()
        for location in visited_by_other:
            if location not in user_visited_locations:
                recommendations[location] += similarity * user_item_matrix.loc[other_user, location]

    ranked_recommendations = sorted(recommendations, key=recommendations.get, reverse=True)[:top_n]
    return ranked_recommendations

# --- 3. Hybrid Approach (Simple Weighted Average) ---
def get_hybrid_recommendations(user_id, users_df, locations_df, location_tfidf_df, user_item_matrix, user_similarity_df, weight_content=0.5, weight_collaborative=0.5, top_n=5):
    content_based_recs = get_content_based_recommendations(user_id, users_df, locations_df, location_tfidf_df, top_n=10)
    collaborative_recs = get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, top_n=10)

    hybrid_scores = defaultdict(float)

    for loc_id in content_based_recs:
        hybrid_scores[loc_id] += weight_content

    for loc_id in collaborative_recs:
        hybrid_scores[loc_id] += weight_collaborative

    ranked_recommendations = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_n]
    return ranked_recommendations

# --- 4. Evaluate ---
train_interactions, test_interactions = train_test_split(interactions_df, test_size=0.2, random_state=42)

def get_relevant_locations(user_id, test_data):
    return test_data[test_data['user_id'] == user_id]['location_id'].unique().tolist()

def precision_at_k(recommended_items, relevant_items, k=5):
    if not recommended_items:
        return 0.0
    top_k = recommended_items[:k]
    relevant_in_top_k = [item for item in top_k if item in relevant_items]
    return len(relevant_in_top_k) / k

def recall_at_k(recommended_items, relevant_items, k=5):
    if not relevant_items:
        return 0.0
    top_k = recommended_items[:k]
    relevant_in_top_k = [item for item in top_k if item in relevant_items]
    return len(relevant_in_top_k) / len(relevant_items)

k_value = 5
hybrid_precisions = []
hybrid_recalls = []

train_user_item_matrix = train_interactions.pivot_table(index='user_id', columns='location_id', values='visit_count').fillna(0)
train_user_similarity_matrix = cosine_similarity(train_user_item_matrix)
train_user_similarity_df = pd.DataFrame(train_user_similarity_matrix, index=train_user_item_matrix.index, columns=train_user_item_matrix.index)

test_users = test_interactions['user_id'].unique()
for user_id in test_users:
    if user_id in train_user_item_matrix.index:
        relevant_locations = get_relevant_locations(user_id, test_interactions)
        recommended_hybrid = get_hybrid_recommendations(
            user_id, users_df, locations_df, location_tfidf_df,
            train_user_item_matrix, train_user_similarity_df, top_n=k_value
        )
        hybrid_precisions.append(precision_at_k(recommended_hybrid, relevant_locations, k_value))
        hybrid_recalls.append(recall_at_k(recommended_hybrid, relevant_locations, k_value))

print("\n--- Evaluation with Actual Locations ---")
if hybrid_precisions:
    print(f"Hybrid Recommendation - Precision@{k_value}: {np.mean(hybrid_precisions):.4f}")
    print(f"Hybrid Recommendation - Recall@{k_value}: {np.mean(hybrid_recalls):.4f}")
else:
    print("No hybrid recommendations could be evaluated for users in the training set.")