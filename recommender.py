# recommender.py
# Week 2 - Task 3: Collaborative Filtering Movie Recommender (SVD)

import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 1. Load MovieLens dataset
ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating
movies = pd.read_csv("movies.csv")    # movieId, title

# 2. Prepare dataset for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

# 3. Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# 4. Train SVD model
model = SVD()
model.fit(trainset)

# 5. Evaluate
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print("✅ Model trained. RMSE:", rmse, "MAE:", mae)

# 6. Save model + movie data
joblib.dump(model, "movie_model.pkl")
movies.to_csv("movies_lookup.csv", index=False)
print("Model saved as movie_model.pkl, movies saved as movies_lookup.csv")
