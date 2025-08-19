import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the pre-tained model with RELU activation
model = load_model('simple_rnn_imdb.h5', compile=False)


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}



# function to preprocess user input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## predicion function 
def predict_sentiment(review):
    prerprocessed_input = preprocess_text(review)
    prediction = model.predict(prerprocessed_input)
    if prediction[0][0] > 0.6:
       
        sentiment = 'Positive'
    # elif 0.5 <= prediction[0][0] <= 0.6:
    #     sentiment = 'Average'
    else:
        sentiment = 'Negative'
    
    return sentiment, prediction[0][0]

## Desigining the stramlit app
import streamlit as st 

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify its **Positive**, **Negative**, or **Average** sentiment.")


# USER INPUT
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
   # Make prediction 
    prediction = model.predict(preprocessed_input)
    
    if prediction[0][0] > 0.5:   # keep same as function
        sentiment = 'Positive'
    # elif prediction[0][0]>0.5 and prediction [0][0]<0.61:
    #     sentiment = 'Average'
    else:
        sentiment = 'Negative'
    
    st.write(f"Prediction: **{sentiment}** (score: {prediction[0][0]:.2f})")

