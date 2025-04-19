import joblib
from preprocess import clean_text

def predict_sentiment(news_text):
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    cleaned = clean_text(news_text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)
    return "UP" if prediction[0] == 1 else "DOWN"

# Example
example_news = "Stocks surged after the Federal Reserve's interest rate decision."
print(f"Prediction: {predict_sentiment(example_news)}")
