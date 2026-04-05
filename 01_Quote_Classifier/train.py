import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("--- Loading Data ---")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scraped_data.csv")

df = pd.read_csv(DATA_PATH)

# 1. Feature Engineering: Turn words into vectors (Math!)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text']) # The input data
y = df['Tag'] # The answers we want to predict

# 2. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model
print("--- Training AI Model ---")
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Test the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%\n")

# 5. Let's test it on entirely new data!
print("--- Live Prediction Test ---")
new_sentences = [
    "A day without sunshine is like, you know, night.",
    "The universe is expanding and physics is amazing."
]

# Convert the new text to math using our trained vectorizer
new_math = vectorizer.transform(new_sentences)
results = model.predict(new_math)

for sentence, prediction in zip(new_sentences, results):
    print(f"Text: '{sentence}'")
    print(f"AI Predicts Category: -> [{prediction}]\n")

import joblib

# Save the trained model and the vectorizer to your hard drive
joblib.dump(model, 'quote_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("--- AI Brain Saved to Disk! ---")
