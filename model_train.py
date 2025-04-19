from preprocess import load_and_preprocess, get_tfidf_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and preprocess data
df = load_and_preprocess("StockNews.csv")
X, y, tfidf = get_tfidf_features(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
