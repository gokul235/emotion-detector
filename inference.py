# inference.py

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Paths
MODEL_PATH = "bert_emotion_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Manually define the label map
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def predict_emotion(text: str) -> str:
    """
    Predict the emotion of the input text using the fine-tuned DistilBERT model.

    Args:
        text (str): Input sentence

    Returns:
        str: Predicted emotion label (e.g., "joy", "anger", etc.)
    """
    inputs = tokenizer(
        text,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=128
    )

    logits = model(inputs).logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return label_map.get(predicted_class, "unknown")


# 🔍 Standalone test
if __name__ == "__main__":
    sample_text = "I'm feeling really good about today!"
    try:
        emotion = predict_emotion(sample_text)
        print(f"✅ Predicted Emotion: {emotion}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

