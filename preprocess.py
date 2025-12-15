import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 500

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def preprocess_text(text, tokenizer):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
