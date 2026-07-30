# >> Twitter Sentiment Analyzer

A lightweight, Flask-based Machine Learning web application that analyzes text and predicts whether the underlying sentiment is Positive or Negative. The project features a custom-built, animated cyber-grid UI and processes text using Natural Language Processing (NLP) techniques.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorization)
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML5, CSS3 (Pure CSS animations, no external assets)

## 🧠 Model Architecture
The model was trained on a subset of the **Sentiment140** dataset (1.6 million tweets). 
* **Text Vectorization:** Utilizes `TfidfVectorizer` with a tuned vocabulary of 10,000 maximum features and unigram/bigram extraction.
* **Classification:** A Logistic Regression model predicts the binary outcome, outputting a probability confidence score mapped to a custom UI threshold.

## 🔬 Known Limitations & Future Scope
Building this prototype highlighted several crucial realities of traditional NLP approaches:
1. **The "Bag of Words" Flaw:** Because TF-IDF processes text as isolated frequencies rather than sequential language, the model struggles with modifier words. For example, adding the word "extremely" to a negative sentence can mathematically flip the prediction to positive due to the high positive weight of the modifier.
2. **Dataset Bias:** The original 2009 Sentiment140 dataset was labeled automatically using emoticons (e.g., `:)` = Positive). Consequently, the model learned internet slang and sarcastic associations rather than pure grammatical English.
3. **Future Upgrades:** Transitioning from Logistic Regression to a Transformer-based architecture (like BERT) or implementing robust text pre-processing (Stemming/Lemmatization) would solve the contextual awareness issues.

