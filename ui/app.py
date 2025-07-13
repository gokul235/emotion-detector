import streamlit as st
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle

@st.cache_resource
def load_model():
    model = TFDistilBertForSequenceClassification.from_pretrained("bert_emotion_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_emotion_model")
    with open("bert_emotion_model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Load model and tokenizer
model, tokenizer, label_encoder = load_model()

# Define label map
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Streamlit UI
st.title("😄 Emotion Detection App")
st.write("Enter a sentence, and the model will predict the emotion.")

user_input = st.text_input("Enter your text here:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="tf", padding=True, truncation=True)
    outputs = model(inputs)
    predicted_class = int(tf.argmax(outputs.logits, axis=1).numpy()[0])
    emotion = label_map[predicted_class]
    st.success(f"🔍 **Predicted Emotion:** {emotion.capitalize()}")

