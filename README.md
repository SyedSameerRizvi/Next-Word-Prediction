# Next Word Prediction Using LSTM and GRU
This project demonstrates next word prediction using two deep learning models: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit). The models are built using TensorFlow and deployed as a web application using Streamlit.

## Installation
Install the required packages:
  ```bash
     pip install -r requirements.txt
  ```
This will install dependencies like TensorFlow, Pandas, NumPy, Streamlit, and more, as listed in the requirements.txt file.

## Usage
To check and predict the next words yourself click on the link: [Next Word Prediction Website](https://next-word-prediction-lx2a7q72vfy226ack4d5uf.streamlit.app/)

## Models

### LSTM Model
- File: `next_word_lstm.keras`
- The LSTM model is used to predict the next word based on a given sequence of words.
### GRU Model
- File: `next_word_gru.keras`
- The GRU model is another recurrent neural network (RNN) architecture used for next word prediction.

## Files
* `requirements.txt`: A file listing all the dependencies required to run the project​(requirements).
* `next_word_lstm.keras`: Pre-trained LSTM model for next word prediction.
* `next_word_gru.keras`: Pre-trained GRU model for next word prediction.
* `tokenizer.pkl`: Tokenizer used to preprocess the input sequences for both models
