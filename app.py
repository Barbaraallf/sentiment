import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load your trained model
model = joblib.load("finalized_model.pkl")

def clean_text(text):
    """Clean the input text by removing punctuation and converting to lowercase."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def predict_sentiment(input_text):
    """Predict the sentiment of the input text and return both prediction and confidence score."""
    cleaned_text = clean_text(input_text)
    prediction = model.predict([cleaned_text])
    confidence = max(model.predict_proba([cleaned_text])[0])  # Assuming your model has predict_proba
    return prediction[0], confidence

# Streamlit app enhancements for a more advanced UI
st.title('Sentiment Analysis with NLP')

with st.form("user_input_form"):
    user_input = st.text_area("Type your message here", height=150)
    submit_button = st.form_submit_button("Predict Sentiment")

if submit_button and user_input:
    sentiment, confidence = predict_sentiment(user_input)
    if sentiment == 1:
        st.success(f'Positive Sentiment 🙂 with confidence {confidence:.2f}')
    else:
        st.error(f'Negative Sentiment 😞 with confidence {confidence:.2f}')

    # Advanced analysis (optional)
    if st.checkbox('Show detailed analysis'):
        st.write('Detailed analysis not implemented. Could include word importance, etc.')
