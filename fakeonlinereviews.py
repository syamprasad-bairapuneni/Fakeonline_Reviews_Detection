import pandas as pd
import re
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import requests

# Step 1: Generate a larger dataset and save to 'reviews_large.csv'
def generate_large_dataset(file_path='reviews_large.csv'):
    reviews = [
        "Great product, highly recommend!",
        "Worst purchase ever. Don't buy.",
        "Average quality, okay for the price.",
        "Excellent service and fast shipping!",
        "Not as described, very disappointed.",
        "Fantastic! Will buy again.",
        "Terrible experience, never again.",
        "Good value for money.",
        "Poor quality, broke after a week.",
        "Amazing, exceeded expectations."
    ]
    labels = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    data = {
        'review': [random.choice(reviews) for _ in range(1000)],
        'label': [random.choice(labels) for _ in range(1000)]
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    print(f"Generated a large dataset with 1000 reviews and saved to '{file_path}'.")

# Step 2: Load the dataset
def load_dataset(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Step 3: Data cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Step 4: Apply data cleaning
def preprocess_data(df):
    df['clean_review'] = df['review'].apply(clean_text)
    return df

# Step 5: TF-IDF Vectorization
def vectorize_data(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_review']).toarray()
    y = df['label']
    return X, y, tfidf

# Step 6: Splitting data and training the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))
    return model

# Step 7: Predicting function
def predict_review(model, tfidf, review):
    clean = clean_text(review)
    vectorized = tfidf.transform([clean]).toarray()
    prediction = model.predict(vectorized)
    return 'Fake' if prediction == 1 else 'Genuine'

# Step 8: Function to fetch real-time reviews
def fetch_real_time_reviews(url):
    response = requests.get(url)
    reviews = response.json()
    return reviews

# Example usage
file_path = 'reviews_large.csv'
generate_large_dataset(file_path)
df = load_dataset(file_path)

if df is not None:
    df = preprocess_data(df)
    X, y, tfidf = vectorize_data(df)
    model = train_model(X, y)

    # Example prediction with a sample review
    review = "This product is amazing!"
    print(predict_review(model, tfidf, review))

    # Using mock data for real-time reviews
    real_time_reviews = [{"text": "Loved this product!"}, {"text": "Not worth the money."}]

    for review in real_time_reviews:
        prediction = predict_review(model, tfidf, review['text'])
        print(f"Review: {review['text']}")
        print(f"Prediction: {prediction}")
