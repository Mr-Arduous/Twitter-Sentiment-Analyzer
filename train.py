import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("Loading dataset...")

# 1. Load the CSV file
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)

# 2. Name the columns
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'tweet']

# 3. Map the target numbers to text labels (0 = Negative, 4 = Positive)
sentiment_map = {0: "Negative", 4: "Positive"}
df['sentiment'] = df['target'].map(sentiment_map)

# 4. Grab 200,000 tweets for a solid middle-ground training size
df_sample = df.sample(n=200000, random_state=42)

# 5. Separate Inputs (X) and Outputs (y)
X = df_sample['tweet']       
y = df_sample['sentiment']   

print("Vectorizing text...")
# 6. Set up the TF-IDF Vectorizer with 10,000 features
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

print("Training the model...")
# 7. Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000) 
model.fit(X_vectorized, y)

# 8. Export the trained model and the vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training complete! Brain expanded to 10k words.")