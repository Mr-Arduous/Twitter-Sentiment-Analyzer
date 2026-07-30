import pickle

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_sentiment(user_text):
    text_vectorized = vectorizer.transform([user_text])
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Convert decimals to percentages (e.g., 0.55 -> 55.0)
    negative_prob = round(probabilities[0] * 100, 1)
    positive_prob = round(probabilities[1] * 100, 1)
    
    # Shrink the Neutral zone by lowering the threshold to 52%
    if positive_prob > 52:
        sentiment = "Positive"
    elif negative_prob > 52:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return sentiment, positive_prob, negative_prob