# Enhanced Travel Recommender App with Nature-Inspired UI

import streamlit as st
import logging
import os
from PIL import Image

from src.data_loader import load_all_data
from src.feature_engineering import prepare_location_features
from src.content_filtering import create_user_preference_vector, get_content_based_recommendations
from src.collaborative_filtering import create_user_location_matrix, get_top_k_similar_users, predict_ratings_for_user
from src.utils import estimate_location_cost, get_user_budget_limit
from src.hybrid import combine_scores
from src import config

st.set_page_config(page_title="Offbeat Oasis - Travel Recommender", layout="wide", page_icon="🌿")

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

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("==== Travel Recommender App Started ====")

# Custom CSS to style the app
def load_css():
    st.markdown("""
    <style>
        /* Main background and text */
        .main {
            background: linear-gradient(135deg, #e6f5e6, #c8e6c9, #a5d6a7);
        }
        .stApp {
            background: linear-gradient(135deg, #e6f5e6, #c8e6c9, #a5d6a7);
        }
        
        /* Header styling */
        .header {
            font-family: "Times New Roman", Times, serif;
            color: #2e7d32;
            text-align: center;
            font-size: 3.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #e8f5e9, #c8e6c9);
            border-right: 1px solid #81c784;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4caf50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
        }
        
        .stButton>button:hover {
            background-color: #388e3c;
            color: white;
        }
        
        /* Slider styling */
        .stSlider .thumb {
            background-color: #2e7d32 !important;
        }
        
        .stSlider .track {
            background: linear-gradient(90deg, #a5d6a7, #2e7d32) !important;
        }
        
        /* Selectbox styling */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #e8f5e9;
            border-color: #81c784;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #81c784;
            border-radius: 8px;
        }
        
        /* Custom card styling for recommendations */
        .recommendation-card {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #2e7d32;
        }
        
        .recommendation-title {
            color: #1b5e20;
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        
        .recommendation-detail {
            color: #2e7d32;
            margin-bottom: 5px;
        }
        
        .recommendation-score {
            color: #4caf50;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Header
st.markdown('<div class="header">Offbeat Oasis</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #2e7d32; margin-bottom: 2em;">Discover Your Perfect Nature Escape</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    logger.info("Loading data...")
    data = load_all_data()
    logger.info("Data loaded successfully.")
    return data

locations_df, trips_df, users_df, reviews_df = load_data()

# Sidebar user input
with st.sidebar:
    st.markdown("""
    <div style="background-color: #2e7d32; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="color: white; text-align: center;">User Preferences</h3>
    </div>
    """, unsafe_allow_html=True)
    
    user_id = st.selectbox("Select User ID", users_df["user_id"].unique())
    
    # User Metadata for personalized content filtering
    user_metadata = users_df[users_df["user_id"] == user_id].iloc[0]
    category = st.selectbox("Preferred Travel Category", locations_df["category"].dropna().unique())
    state = st.selectbox("Preferred State", locations_df["state"].dropna().unique())
    top_k = st.slider("Number of Recommendations", 1, 20, config.TOP_K)
    
    # Budget Mode Selection
    st.markdown("---")
    st.markdown("#### Budget Preferences")
    budget_mode = st.radio("Budget Mode", ["Auto (from profile)", "Manual"], index=0)
    if budget_mode == "Manual":
        user_budget = st.slider("Trip Budget (₹)", 10000, 200000, 50000, step=5000)
    else:
        user_budget = get_user_budget_limit(users_df, user_id)
    
    st.markdown(f"""
    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 8px; margin-top: 20px;">
        <p style="color: #2e7d32; font-weight: bold;">Selected Budget: ₹{user_budget:,}</p>
    </div>
    """, unsafe_allow_html=True)

logger.info(f"User selected: user_id={user_id}, category='{category}', state='{state}', budget_mode={budget_mode}, budget=₹{user_budget}, top_k={top_k}")

# Feature engineering
logger.info("Preparing location features using TF-IDF and numeric data...")
tfidf_matrix, tfidf = prepare_location_features(locations_df)

user_vector = create_user_preference_vector(category, state, tfidf, {
    "occupation": user_metadata["occupation"],
    "location_type": user_metadata["location_type"]
})

content_recs = get_content_based_recommendations(user_vector, tfidf_matrix, locations_df, top_n=50)
logger.info(f"Generated {len(content_recs)} content-based recommendations.")

# Collaborative filtering
logger.info("Running collaborative filtering...")
interaction_matrix = create_user_location_matrix(reviews_df, users_df)
similar_users = get_top_k_similar_users(interaction_matrix, user_id, k=5)
collab_scores = predict_ratings_for_user(interaction_matrix, similar_users, user_id)
logger.info("Collaborative filtering scores calculated.")

# Estimate costs and filter by user's budget
logger.info("Estimating costs and applying budget filter...")
location_costs = estimate_location_cost(reviews_df, trips_df)
content_recs = content_recs.merge(location_costs, on="location_id", how="left")
filtered_recs = content_recs[content_recs["estimated_cost"] <= user_budget]
logger.info(f"{len(filtered_recs)} recommendations remain after budget filtering (limit: ₹{user_budget}).")

# Combine scores (Hybrid)
final_recs = combine_scores(
    filtered_recs,
    collab_scores,
    user_id,
    reviews_df,
    weight_content=config.WEIGHT_CONTENT,
    weight_collab=config.WEIGHT_COLLAB
)

logger.info(f"{len(final_recs)} final recommendations after hybrid scoring.")

# Display recommendations
st.markdown("""
<div style="background-color: #2e7d32; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
    <h2 style="color: white; text-align: center;">🌿 Top Recommended Locations for You</h2>
</div>
""", unsafe_allow_html=True)

if not final_recs.empty:
    # Get top recommendation for featured display
    top_recommendation = final_recs.iloc[0]
    
    # Determine which image to display based on category
    category_images = {
        'remote_nature': 'remote_image.jpg',
        'adventure_hiking': 'remote_image.jpg',
        'hill_station_trails': 'remote_image.jpg',
        'local_getaway': 'local_getaway.jpg',
        'beach_retreat': 'beach_images.jpg',
        'remote_nature_religious': 'religion.jpg',
        'wildlife_ecotourism': 'eco.jpg',
        'eco_adventure': 'eco.jpg',
        'cultural_escape': 'tribals.jpg',
        'tribal_retreat': 'tribals.jpg'
    }
    
    image_file = category_images.get(top_recommendation['category'], 'common.jpg')
    
    try:
        image = Image.open(f"images/{image_file}")
    except:
        # Fallback if image not found
        image = Image.open("images/common.jpg")
    
    # Create two columns for the featured recommendation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, use_column_width=True, caption=top_recommendation['location_name'])
    
    with col2:
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="recommendation-title">{top_recommendation['location_name']}</div>
            <div class="recommendation-detail"><strong>State:</strong> {top_recommendation['state']}</div>
            <div class="recommendation-detail"><strong>Category:</strong> {top_recommendation['category'].replace('_', ' ').title()}</div>
            <div class="recommendation-detail"><strong>Estimated Cost:</strong> ₹{top_recommendation['estimated_cost']:,.0f}</div>
            <div class="recommendation-detail"><strong>Match Score:</strong> <span class="recommendation-score">{top_recommendation['hybrid_score']:.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display remaining recommendations in a styled table
    if top_k > 1:
        st.markdown("### Other Recommendations")
        styled_df = final_recs[["location_name", "state", "category", "estimated_cost", "hybrid_score"]].head(top_k).copy()
        styled_df['estimated_cost'] = styled_df['estimated_cost'].apply(lambda x: f"₹{x:,.0f}")
        styled_df['hybrid_score'] = styled_df['hybrid_score'].apply(lambda x: f"{x:.2f}")
        styled_df.columns = ['Location', 'State', 'Category', 'Estimated Cost', 'Match Score']
        
        # Apply custom styling to the dataframe
        st.dataframe(
            styled_df,
            column_config={
                "Location": st.column_config.TextColumn(width="large"),
                "State": st.column_config.TextColumn(width="medium"),
                "Category": st.column_config.TextColumn(width="medium"),
                "Estimated Cost": st.column_config.TextColumn(width="medium"),
                "Match Score": st.column_config.ProgressColumn(
                    "Match Score",
                    help="How well this matches your preferences",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    logger.info("Recommendations displayed successfully.")
else:
    st.markdown("""
    <div style="background-color: #ffebee; padding: 20px; border-radius: 8px; border-left: 5px solid #f44336;">
        <h3 style="color: #c62828;">No Recommendations Found</h3>
        <p style="color: #c62828;">We couldn't find any locations that match your preferences and budget. Try adjusting your filters.</p>
    </div>
    """, unsafe_allow_html=True)
    logger.warning("No recommendations generated for the current user input.") 

