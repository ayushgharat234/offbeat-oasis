
# 🌍 Offbeat Oasis — Travel Recommendation System

A hybrid travel recommender engine that suggests personalized travel destinations to users based on preferences, behavior, and budget.

---

## 🧠 Overview

This system combines **Content-Based Filtering** and **Collaborative Filtering**, and intelligently merges the results via **Hybrid Scoring** to generate top travel destination recommendations for a user.

---

## 📁 Project Structure

```
offbeat-oasis/
├── app/
│   ├── app.py           # Main Streamlit application
│   └── app2.py          # Alternate/experimental Streamlit version
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── content_filtering.py
│   ├── collaborative_filtering.py
│   ├── hybrid.py
│   ├── utils.py
│   ├── config.py
│   └── __init__.py
└── data/final/
    ├── locations.csv
    ├── users.csv
    ├── trips.csv
    └── reviews.csv
```

---

## 📦 Data Files

### `locations.csv`
- `location_id`, `location_name`, `category`, `state`
- `activities`, `places`, `num_activities`, `num_places`

### `users.csv`
- `user_id`, `occupation`, `location_type`
- Budget flags: `budget_under_25k`, `budget_25k_to_50k`, `budget_50k_to_100k`, `budget_above_100k`

### `reviews.csv`
- `user_id`, `location_id`, `rating`

### `trips.csv`
- `user_id`, `location_id`, `cost`

---

## 🔧 Feature Engineering

Location metadata is processed into a TF-IDF matrix using:
- Textual features: `category`, `state`, `activities`, `places`
- Normalized numeric features: `num_activities`, `num_places`

Each location is converted into a vector:
```python
[TF-IDF terms] + [scaled activity count] + [scaled place count]
```

---

## 🎯 Recommendation Engine

### 1. Content-Based Filtering

**User Vector** is built using:
- Selected category, state, occupation, and location type

TF-IDF similarity is computed between user vector and each location vector.

```python
similarity = cosine_similarity(user_vector, tfidf_matrix)
```

---

### 2. Collaborative Filtering

Creates a `user_id x location_id` matrix based on ratings.

- Computes cosine similarity between users
- Predicts ratings for locations not yet rated by the user

```python
predicted_ratings = weighted_average_from_similar_users()
```

---

### 3. Hybrid Recommendation

Scores are normalized and combined using weighted average:

```python
hybrid_score = weight_content * normalized_content_score + weight_collab * normalized_collab_score
```

Dynamic weighting based on user activity:
- <3 reviews → prioritize content (0.8/0.2)
- >10 reviews → prioritize collaborative (0.3/0.7)

---

### 4. Budget Filtering

Each user's budget is inferred from flag fields. Locations are filtered out if:

```python
estimated_cost > user_budget_limit
```

---

## ⚙️ Configuration (`src/config.py`)

```python
WEIGHT_CONTENT = 0.6
WEIGHT_COLLAB = 0.4
TOP_K = 10
DEFAULT_TRIP_BUDGET = 10000
```

---

## 🖥️ Running the App

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn
```

### 2. Run Streamlit

```bash
streamlit run app/app.py
```

Make sure CSVs are in `data/final/`.

---

## 📈 Output

Displays Top-K travel destination recommendations with:

- Name
- State
- Category
- Estimated Cost
- Hybrid Score

---

## ✅ Example UI

Users select:
- Travel category
- Preferred state
- Budget (inferred from profile)
- Number of top recommendations

---

## 🧪 Testing Suggestions

- Try users with/without review history
- Adjust budget filters
- Validate with extreme `Top K` values

---

## 📚 Future Improvements

- Add image previews of destinations
- Integrate map-based visualization
- Include weather or seasonal recommendations
- User input-based dynamic numeric feature customization

---

Built with ❤️ by the Offbeat Oasis Team.
