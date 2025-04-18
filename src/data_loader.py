import pandas as pd

def load_all_data():
    locations_df = pd.read_csv("data/final/locations.csv")
    trips_df = pd.read_csv("data/final/trips.csv")
    users_df = pd.read_csv("data/final/users.csv")
    reviews_df = pd.read_csv("data/final/reviews.csv")
    return locations_df, trips_df, users_df, reviews_df