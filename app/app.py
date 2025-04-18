# travel_app.py (Main Streamlit App)

import streamlit as st
import logging
import os

from src.data_loader import load_all_data
from src.feature_engineering import prepare_location_features
from src.content_filtering import create_user_preference_vector, get_content_based_recommendations
from src.collaborative_filtering import create_user_location_matrix, get_top_k_similar_users, predict_ratings_for_user
from src.utils import estimate_location_cost, apply_budget_filter
from src.hybrid import combine_scores

# Setup robust logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "app.log")

logger = logging.getLogger("travel_app_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Optional: also log to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("==== Travel Recommender App Started ====")

st.set_page_config(page_title="Travel Recommender", layout="wide")
st.title("\U0001F304 Offbeat Oasis - Travel Recommender")

# Load data
@st.cache_data
def load_data():
    logger.info("Loading data...")
    data = load_all_data()
    logger.info("Data loaded successfully.")
    return data

locations_df, trips_df, users_df, reviews_df = load_data()

# Sidebar user input
st.sidebar.header("User Preferences")
user_id = st.sidebar.selectbox("Select User ID", reviews_df["user_id"].unique())
category = st.sidebar.selectbox("Preferred Travel Category", locations_df["category"].dropna().unique())
state = st.sidebar.selectbox("Preferred State", locations_df["state"].dropna().unique())
budget = st.sidebar.slider("Trip Budget (₹)", 10000, 100000, 40000, step=5000)
top_k = st.sidebar.slider("Top K Recommendations", 1, 20, 10)

logger.info(f"User selected: user_id={user_id}, category='{category}', state='{state}', budget=₹{budget}, top_k={top_k}")

# Feature engineering
logger.info("Preparing location features using TF-IDF...")
tfidf_matrix, tfidf = prepare_location_features(locations_df)
user_vector = create_user_preference_vector(category, state, tfidf)
content_recs = get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=50)
logger.info(f"Generated {len(content_recs)} content-based recommendations.")

# Collaborative filtering
logger.info("Running collaborative filtering...")
interaction_matrix = create_user_location_matrix(reviews_df)
similar_users = get_top_k_similar_users(interaction_matrix, user_id, k=5)
collab_scores = predict_ratings_for_user(interaction_matrix, similar_users, user_id)
logger.info("Collaborative filtering scores calculated.")

# Budget filtering
logger.info("Estimating costs and applying budget filter...")
location_costs = estimate_location_cost(reviews_df, trips_df)
filtered_recs = apply_budget_filter(content_recs, location_costs, budget)
logger.info(f"{len(filtered_recs)} recommendations remain after budget filtering.")

# Combine scores (Hybrid)
final_recs = combine_scores(filtered_recs, collab_scores, weight_content=0.6, weight_collab=0.4)
logger.info(f"{len(final_recs)} final recommendations after hybrid scoring.")

# Display recommendations
st.subheader("\U0001F3AF Top Recommended Locations for You")
if not final_recs.empty:
    st.dataframe(final_recs[["location_name", "state", "category", "estimated_cost", "hybrid_score"]].head(top_k))
    logger.info("Recommendations displayed successfully.")
else:
    st.warning("No recommendations match your preferences and budget.")
    logger.warning("No recommendations generated for the current user input.")
