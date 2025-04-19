# 📰 Stock Market Prediction Using News Headlines

This project uses historical news headlines to predict the next day's stock market direction using machine learning.

## Dataset

- [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews)
- Columns: `Label` (0=Down, 1=Up), `Top1` to `Top25` (News headlines)

## Project Structure

- `preprocess.py`: Clean and vectorize news
- `model_train.py`: Train logistic regression model
- `predict.py`: Make new predictions

## How to Use

1. Download the dataset and rename it to `StockNews.csv` in the same folder.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the model:
```bash
python model_train.py
```
4. Run a prediction:
```bash
python predict.py
```

## Model

- Logistic Regression with TF-IDF vectorizer
- Accuracy around 85% depending on seed
