import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]  # Trim to fit input size
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "..."

# Streamlit App UI
st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.markdown("## 📖 Next Word Predictor using LSTM")
st.markdown("Enter a short phrase and predict the next word(s) using a trained LSTM model.")

# Input Section
input_text = st.text_input("📝 Input Text", value="To be or not to", max_chars=100)
num_words = st.slider("🔁 How many words to predict?", 1, 10, 1)

# Prediction Logic
if st.button("🚀 Predict"):
    max_sequence_len = model.input_shape[1] + 1
    output_text = input_text.strip()
    with st.spinner("Generating..."):
        for _ in range(num_words):
            next_word = predict_next_word(model, tokenizer, output_text, max_sequence_len)
            output_text += " " + next_word

    st.success("✅ Prediction Complete!")
    st.markdown("### ✨ Completed Sentence:")
    st.markdown(f"**{output_text}**")

# Sidebar info
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("This app uses a pre-trained LSTM model to generate text predictions. You can enter a phrase and choose how many next words you want to predict.")
