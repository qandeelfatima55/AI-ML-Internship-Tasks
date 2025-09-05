# predict.py
# Load saved model and predict new SMS messages

import joblib

# Load trained model + vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_message(msg):
    msg_tfidf = vectorizer.transform([msg])
    pred = model.predict(msg_tfidf)[0]
    return "Spam" if pred == 1 else "Not Spam"

# Example predictions
print(classify_message("Congratulations! You have won a free ticket!"))
print(classify_message("Let's meet for lunch tomorrow."))
