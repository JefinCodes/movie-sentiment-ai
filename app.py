import streamlit as st
import joblib

# Load saved model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a movie review and I will tell you if it's Positive or Negative!")

# Text input from user
review = st.text_area("Enter your movie review here:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please type a review first.")
    else:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")
