import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset
dataset_path = "/content/drive/My Drive/twi.csv"  # Path in Google Colab

data = pd.read_csv(dataset_path)

# Separate features (text) and labels (sentiment)
texts = data['text'].values
labels = data['sentiment'].values

# Tokenize and pad sequences
vocab_size = 10000  
max_length = 50  
oov_token = "<OOV>"  

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

# Convert texts to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # Number of epochs
batch_size = 32  # Batch size

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Save the model
model.save("twitter_sentiment_model.h5")

# Test with custom text
custom_text = ["fuck you"]
custom_seq = tokenizer.texts_to_sequences(custom_text)
custom_padded = pad_sequences(custom_seq, maxlen=max_length, padding='post')
custom_prediction = model.predict(custom_padded)

if custom_prediction[0][0] > 0.5:
    print("Sentiment: Happy")
else:
    print("Sentiment: Sad")

