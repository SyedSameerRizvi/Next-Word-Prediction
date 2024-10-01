import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.keras')

#3 Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]  # Ensure the sequence length matches max_sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM")
input_text=st.text_input("Enter the sequence of Words","Barn. In the same figure, like the")
if st.button("Predict Next Word"):
    max_sequence = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence)
    st.write(f'Next word: {next_word}')

#Load the GRU Model
model=load_model('next_word_gru.keras')

#3 Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Create a Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence:
        token_list = token_list[-(max_sequence):]  # Ensure the sequence length matches max_sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction With GRU")
input_text=st.text_input("Enter the sequence of Words","Therefore our sometimes Sister, now our")
if st.button("Predict"):
    max_sequence = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence)
    st.write(f'Next word: {next_word}')