# recommend.py
# Load saved model and recommend movies for a user

import pandas as pd
import joblib

# Load model and movie data
model = joblib.load("movie_model.pkl")
movies = pd.read_csv("movies_lookup.csv")

def recommend_movies(user_id, n=5):
    # Predict rating for each movie
    predictions = []
    for movie_id in movies['movieId'].unique():
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    # Sort by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top n movies
    top_movies = [movies[movies['movieId']==mid]['title'].values[0] for mid, _ in predictions[:n]]
    return top_movies

# Example usage
print(recommend_movies(1, 5))  # Top 5 for user 1
