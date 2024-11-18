#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# Load the JSON content
with open(r'C:\Users\Ravi-fms\Desktop\content1\intents (1).json') as content:
    data1 = json.load(content)


# Prepare inputs and tags
tags = []
inputs = []
responses = {}

for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

# Create DataFrame
data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Text preprocessing (removing punctuation, lowering case)
import string
data['inputs'] = data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

# Tokenizing the text
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

# Padding sequences
x_train = pad_sequences(train)

# Label encoding for tags
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

# Define model parameters
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

# Model Definition
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Define the model
i = Input(shape=(input_shape,))
x = Embedding(input_dim=vocabulary + 1, output_dim=10)(i)
x = LSTM(10)(x)
x = Dense(output_length, activation='softmax')(x)

model = Model(i, x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Training the model
history = model.fit(x_train, y_train, epochs=195, batch_size=32)

# Plotting accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend(['Accuracy', 'Loss'])
plt.show()

# Testing the model (example)
def chat_response(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=input_shape)
    pred = model.predict(padded)
    tag = le.inverse_transform([np.argmax(pred)])

    return np.random.choice(responses[tag[0]])

# Testing a few responses
print(chat_response("Hello"))
print(chat_response("Can you help me?"))
print(chat_response("What time do you close?"))


# In[8]:


print(chat_response("Hello"))


# In[11]:


print(chat_response("store timings?"))


# In[12]:


print(chat_response("tell me a joke"))


# In[13]:


print(chat_response("goat?"))


# In[15]:


print(chat_response("who is goat"))

