from django.shortcuts import render
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

# Constants
MAX_WORDS = 10000 # The maximum number of words to keep, based on word frequency.
MAXLEN = 1000 # The maximum length of sequences (number of words).
MODEL_PATH = 'news_classification_best_model.h5' # File path to save the trained model.
BATCH_SIZE = 64 # Number of samples per gradient update.
EPOCHS = 20 # Number of training epochs.

# Function to preprocess text
def preprocess_text(texts, max_words=MAX_WORDS, maxlen=MAXLEN):
    """
        - Tokenizes and pads the input texts.
        - Tokenizer converts texts to sequences of integers.
        - pad_sequences ensures all sequences have the same length by padding them with zeros.

    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=maxlen)
    return data, tokenizer

# Function to create and train the model
def create_and_train_model():
    """
        - Fetches the training data using fetch_20newsgroups.
        - Preprocesses the text data.
        - Defines a Sequential model with an embedding layer, a dropout layer, a Conv1D layer, a max-pooling layer,
            a bidirectional LSTM layer, and a dense output layer with softmax activation.
        - Compiles the model using Adam optimizer and sparse categorical cross-entropy loss.
        - Uses early stopping to avoid overfitting.
        - Trains the model on the preprocessed data and saves the best model to MODEL_PATH.
        - Plots the training and validation loss using plot_loss.
    """
    # Load data
    newsgroups_train = fetch_20newsgroups(subset='train')
    X_train, y_train = newsgroups_train.data, newsgroups_train.target

    # Preprocess text data
    X_train, tokenizer = preprocess_text(X_train)

    # Define the model architecture
    model = Sequential([
        Embedding(MAX_WORDS, 200, input_length=MAXLEN),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
        Dense(20, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model and tokenizer
    model.save(MODEL_PATH)
    
    # Plot loss
    plot_loss(history)

    return model, tokenizer

# Function to plot training and validation loss

def plot_loss(history):
    """"
        Plots the training and validation loss for each epoch to visualize the model performance.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to load the trained model and tokenizer
def load_trained_model():
    """
        - Checks if the model file exists. If not, trains a new model.  
        - If the model file exists, loads the saved model.
        - Initializes the tokenizer correctly using training data.
    """
    if not os.path.exists(MODEL_PATH):
        model, tokenizer = create_and_train_model()
    else:
        model = load_model(MODEL_PATH)
        newsgroups_train = fetch_20newsgroups(subset='train')
        _, tokenizer = preprocess_text(newsgroups_train.data)  # To initialize the tokenizer correctly

    return model, tokenizer

# Function to predict category
def predict_category(model, tokenizer, news_text):
    """
        - Preprocesses the input news text.
        - Predicts the category probabilities using the trained model.
        - Returns the index of the category with the highest probability.
    """
    # Preprocess the input text
    news_text_seq = tokenizer.texts_to_sequences([news_text])
    news_text_pad = pad_sequences(news_text_seq, maxlen=MAXLEN)
    
    # Predict probabilities for each category
    predicted_probabilities = model.predict(news_text_pad)[0]
    
    # Find the index of the category with the highest probability
    predicted_category_index = predicted_probabilities.argmax()
    
    return predicted_category_index

# View function for news classification
def news_classification(request):
    """
        - Handles HTTP GET requests.
        - Gets the input news text from the request.
        - Loads the trained model and tokenizer.
        - Predicts the category of the input news text.
        - Renders the result in the news_classification.html template.
    """
    news_text = request.GET.get('text')
    category = ''
    if news_text:
        # Load the trained model and tokenizer
        model, tokenizer = load_trained_model()

        # Predict category
        predicted_category_index = predict_category(model, tokenizer, news_text)

        # Load target names
        newsgroups_train = fetch_20newsgroups(subset='train')
        category = newsgroups_train.target_names[predicted_category_index]

    return render(request, 'news/news_classification.html', {'category': category, 'news_text': news_text})
