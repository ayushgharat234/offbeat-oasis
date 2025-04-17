def normalize_scores(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def estimate_location_cost(reviews_df, trips_df):
    merged = reviews_df.merge(trips_df, on="user_id", how="left")
    location_costs = merged.groupby("location_id")["cost"].mean().reset_index()
    location_costs.columns = ["location_id", "estimated_cost"]
    return location_costs

def apply_budget_filter(location_df, location_costs, trip_budget):
    merged = location_df.merge(location_costs, on="location_id", how="left")
    return merged[merged["estimated_cost"] <= trip_budget].sort_values("estimated_cost")