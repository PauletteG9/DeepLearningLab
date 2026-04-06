import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

corpus = [
  "the cat sat on the mat",
  "the dog ran in the park",
  "the bird flew over the tree",
  "the fish swam in the lake",
  "the cat ran over the mat",
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1   # +1 for padding token (index 0)
print(f"Vocabulary size: {total_words}")
print(f"Word index: {tokenizer.word_index}")

input_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_sequences.append(n_gram_seq)

max_seq_len = max(len(s) for s in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X = input_sequences[:, :-1]          
y = input_sequences[:, -1]           
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

rnn_model = Sequential([
  layers.Embedding(total_words, 64, input_length=max_seq_len - 1),  # VIVA: 64-dim word vectors
  layers.SimpleRNN(128, return_sequences=False),   # VIVA: 128 hidden units, outputs last state only
  layers.Dense(total_words, activation='softmax'), # VIVA: softmax over vocab → probability per word
], name="RNN_NextWord")

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn_model.summary()
rnn_model.fit(X, y, epochs=50, verbose=1)

index_word = {v: k for k, v in tokenizer.word_index.items()}  # {1:'the', 2:'cat', ...}

def predict_next_word(model, tokenizer, index_word, seed_text, max_seq_len):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre') 
  predicted  = model.predict(token_list, verbose=0)
  pred_index = np.argmax(predicted)              # VIVA: argmax picks highest-probability word
  return index_word.get(pred_index, "<unknown>") # O(1) dict lookup — efficient

seed = "the cat"
next_word = predict_next_word(rnn_model, tokenizer, index_word, seed, max_seq_len)
print(f"\nSeed: '{seed}' → Predicted next word: '{next_word}'")
