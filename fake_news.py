
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('WELFake_Dataset.csv')

# 🔥 Reduce dataset for speed
df = df.sample(3000, random_state=42)

# Remove missing values
df = df.dropna()

# Combine title + text
df['content'] = df['title'] + " " + df['text']

# Setup stemming
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# 🔥 Show progress
print("⏳ Preprocessing started...")
df['content'] = df['content'].apply(stemming)
print("✅ Preprocessing finished")

# Features & labels
X = df['content'].values
Y = df['label'].values

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

print("\n✅ Model Ready for Prediction")

# 🔮 Prediction loop
while True:
    news = input("\nEnter news (or type 'exit'): ")

    if news.lower() == 'exit':
        break

    processed = stemming(news)
    vector = vectorizer.transform([processed])

    prob = model.predict_proba(vector)[0]
    news_lower = news.lower()

    # 🔴 Fake rules (FINAL)
    is_fake_rule = (
        ("all students" in news_lower and "pass" in news_lower) or
        ("shut down" in news_lower and "internet" in news_lower) or
        ("shut down" in news_lower and "schools" in news_lower) or
        ("cure" in news_lower and "all diseases" in news_lower) or
        ("stop aging" in news_lower) or
        ("free car" in news_lower) or
        ("free for every citizen" in news_lower) or
        ("survive without food" in news_lower) or
        ("aliens" in news_lower) or
        ("free iphone" in news_lower)
    )

    # 🟢 Real hints (FINAL)
    is_real_hint = (
        ("isro" in news_lower) or
        ("satellite" in news_lower) or
        ("meteorological" in news_lower) or
        ("reserve bank" in news_lower) or
        ("railways" in news_lower) or
        ("train" in news_lower) or
        ("renewable energy" in news_lower) or
        ("funding" in news_lower) or
        ("experts say" in news_lower) or
        ("introduced" in news_lower) or
        ("ministry" in news_lower) or
        ("education" in news_lower) or
        ("health" in news_lower) or
        ("election commission" in news_lower) or
        ("launched" in news_lower and "campaign" in news_lower) or
        ("temperature" in news_lower) or
        ("climate" in news_lower) or
        ("global warming" in news_lower)
    )

    # Show probabilities
    print(f"Fake prob: {prob[1]:.2f} | Real prob: {prob[0]:.2f}")

    # ✅ FINAL DECISION
    if is_fake_rule:
        print("🚨 FAKE NEWS (Rule-based detection)")
    elif is_real_hint:
        print("✅ REAL NEWS (Trusted keywords)")
    elif prob[1] > 0.65:
        print("🚨 FAKE NEWS")
    elif prob[0] > 0.65:
        print("✅ REAL NEWS")
    else:
        print("⚠️ UNCERTAIN (Low confidence)")