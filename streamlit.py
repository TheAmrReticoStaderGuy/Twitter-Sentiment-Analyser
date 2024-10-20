from utils import clean_text
import streamlit as st
import pickle
# Streamlit UI

from sklearn.feature_extraction.text import TfidfVectorizer
import time
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2)) 

st.title("Tweet Sentiment Analyzer")

# Text input for the tweet
tweet_input = st.text_area("Enter a tweet:", placeholder="Type your tweet here...")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if tweet_input.strip():
        # Preprocess the tweet
        clean_tweet = clean_text(tweet_input)

        # Vectorize the cleaned tweet
        tweet_vectorized = vectorizer.transform([clean_tweet])

        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Time the prediction
        start_time = time.time()
        prediction = model.predict(tweet_vectorized)[0]
        prediction_time = time.time() - start_time

        # Display the result
        st.write(f"**Predicted Sentiment:** {prediction.capitalize()}")
        st.write(f"**Time Taken for Prediction:** {prediction_time:.3f} seconds")
    else:
        st.write("Please enter a valid tweet.")