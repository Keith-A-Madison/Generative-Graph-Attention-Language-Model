import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense
from spektral.layers import GCNConv
from spektral.utils import normalized_adjacency
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import get_training_data

# Need to change all the 9's as soon as the vocab size grows :(
corpus = get_training_data("/home/kmadison/gpt/GenAtten/training/") # ["The quick brown fox jumps", "over the lazy dog", "and goes to the store", "to buy some tacos"] #, "and goes to the store", "to buy some tacos"]
tokens = [sentence.split() for sentence in corpus]
#tokens.append('<PAD>')
vocab = set(token for sentence in tokens for token in sentence)
#vocab.append('<PAD>')
#print(vocab)
vocab_size = len(vocab)

print("Vocabulary size: ", vocab_size)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Calculate token-to-token similarity matrix
similarity_matrix = np.zeros((len(vocab), len(vocab)))
for sentence in tokens:
    for i, token1 in enumerate(sentence):
        for j, token2 in enumerate(sentence):
            similarity_matrix[word2idx[token1], word2idx[token2]] += 1

# Normalize similarity matrix to get adjacency matrix
adjacency_matrix = normalized_adjacency(similarity_matrix)

# print(adjacency_matrix, vocab_size)

input_layer = Input(shape=(None,))
embedding = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)
gcn_layer = GCNConv(128)([embedding, adjacency_matrix])
output_layer = Dense(vocab_size, activation='softmax')(gcn_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training data

# Convert tokens to indices
train_data = [[word2idx[token] for token in sentence] for sentence in tokens]

# Padding and truncating sequences to a fixed length
# max_sequence_length = max(len(sentence) for sentence in train_data)
# For some reason, max_sequence_length must equal the number of vocabular words???
max_sequence_length = vocab_size
train_data_padded = pad_sequences(train_data, maxlen=max_sequence_length, padding='post', truncating='post')

# Shift train_data_padded to create train_labels
train_labels = np.roll(train_data_padded, -1, axis=1)
train_labels[train_labels >= vocab_size] = 0

print(train_data_padded)

# Create NumPy arrays from train_data_padded and train_labels
train_data_padded = np.array(train_data_padded)
train_labels = np.array(train_labels)

num_epochs = 100
# Training loop
model.fit(train_data_padded, tf.keras.utils.to_categorical(train_labels, num_classes=vocab_size), epochs=num_epochs)

print(vocab_size)

padding_token = "<PAD>"

# Text generation
initial_string = "The traffic"
initial_tokens = initial_string.split()

# Fill up to 9 tokens if the initial string is shorter
while len(initial_tokens) < vocab_size:
    initial_tokens.insert(0, padding_token)
#    initial_tokens.append(padding_token)  # Replace with appropriate padding token

generated_tokens = [word2idx[token] for token in initial_tokens]
max_generation_length = 40

# Number of tokens from the initial string that have been used
initial_tokens_used = len(initial_tokens)

for _ in range(max_generation_length):
    current_seq = np.array([generated_tokens[-vocab_size:]])  # Always use the last vocab_size tokens
    next_token_probs = model.predict(current_seq)[0][-1] # Not sure if this is right?
    next_token = np.random.choice(np.arange(vocab_size), p=next_token_probs)
    generated_tokens.append(next_token)

    # If there are more initial tokens to use, do that
    if initial_tokens_used < len(initial_tokens):
        generated_tokens[-vocab_size:] = [word2idx[token] for token in initial_tokens[initial_tokens_used:]]
        initial_tokens_used = len(initial_tokens)

# Generate text
generated_text = [idx2word[idx] for idx in generated_tokens]

# Remove trailing "<PAD>" tokens
generated_text = [token for token in generated_text if token != padding_token]

# Join the tokens into text
generated_text = " ".join(generated_text)
print("Generated Text:", generated_text)
