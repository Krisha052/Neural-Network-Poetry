"""
Task:
- Building a recurrent neural network that generates texts like Shakespeare's texts
"""

import random
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Activation 
from tensorflow.keras.optimizers import RMSprop 

# Define the filepath to the Shakespeare text file
filepath = '/Users/krisha/Desktop/TextAI/shakespeare.txt'

# Read the text file and decode it
with open(filepath, 'rb') as f:
    text = f.read().decode(encoding='utf-8').lower() 

# Limit the text for training
text = text[300000:800000]

# Create a sorted list of unique characters
characters = sorted(set(text)) 

# Create character-to-index and index-to-character dictionaries
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Define sequence length and step size for training
seq_length = 40
step_size = 3

# Prepare the training data
sentences = [] # target
next_char = [] # features

for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i: i + seq_length])
    next_char.append(text[i + seq_length])

# Use np.bool_ for compatibility with future versions of NumPy
x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

# Populate the input and output data
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Function to sample from the predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # Added small value to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text
def generate_text(model, length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

# Build the model (assuming you have the model defined)
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(characters)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Assuming you have fitted the model before generating text
model.fit(x, y, batch_size=128, epochs=10, verbose=1)  

# Generate text with different temperatures
for temp in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
    print(f"Generated text at temperature {temp}:")
    print(generate_text(model, 300, temp))
