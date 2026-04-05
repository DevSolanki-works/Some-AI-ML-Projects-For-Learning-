import joblib

print("Loading AI Brain...")
model = joblib.load('quote_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("\n--- Quote Classifier Live ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a quote: ")
    if user_input.lower() == 'exit':
        break
        
    # Convert text to math, then predict
    math_input = vectorizer.transform([user_input])
    prediction = model.predict(math_input)[0]
    
    print(f"AI Predicts Tag: -> [{prediction}]\n")
