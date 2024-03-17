import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load your trained model
model = joblib.load("path/to/your_model.pkl")

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def predict_sentiment(input_text):
    cleaned_text = clean_text(input_text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Streamlit app
st.title('Sentiment Analysis with NLP')
user_input = st.text_area("Type your message here")
if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(user_input)
    if sentiment == 1:
        st.write('Positive Sentiment 🙂')
    else:
        st.write('Negative Sentiment 😞')

