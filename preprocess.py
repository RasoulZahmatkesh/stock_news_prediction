import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Combined'] = df.iloc[:, 2:27].astype(str).agg(' '.join, axis=1)
    df['Combined'] = df['Combined'].apply(clean_text)
    df['Label'] = df['Label'].map({0: 0, 1: 1})
    return df[['Combined', 'Label']]

def get_tfidf_features(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Combined']).toarray()
    y = df['Label'].values
    return X, y, tfidf
