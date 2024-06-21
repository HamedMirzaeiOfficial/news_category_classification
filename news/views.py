from django.shortcuts import render
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Bidirectional, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping


# Function to preprocess text
def preprocess_text(texts, max_words, maxlen):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=maxlen)
    return data

# Function to create and train the deep learning model
def create_and_train_model():
    # Load data
    newsgroups_train = fetch_20newsgroups(subset='train')
    X_train = newsgroups_train.data
    y_train = newsgroups_train.target

    max_words = 10000
    maxlen = 1000

    # Preprocess text data
    X_train = preprocess_text(X_train, max_words, maxlen)

    # Define the model architecture
    model = Sequential()
    model.add(Embedding(max_words, 200, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(20, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model with a new postfix
    model.save('news_classification_best_model.h5')

    
# Function to load the trained model
def load_trained_model():
    # Load the trained model
    model = load_model('news_classification_best_model.h5')
    return model

# Function to predict category
def predict_category(model, news_text):
    max_words = 10000
    maxlen = 100
    # Preprocess the input text
    news_text = preprocess_text([news_text], max_words, maxlen)
    # Predict probabilities for each category
    predicted_probabilities = model.predict(news_text)[0]
    # Find the index of the category with the highest probability
    predicted_category_index = predicted_probabilities.argmax()
    return predicted_category_index

# View function for news classification
def news_classification(request):
    news_text = request.GET.get('text')
    category = ''
    if news_text:
        # Check if model is trained and load it
        try:
            model = load_trained_model()
        except FileNotFoundError:
            # If model is not trained, create and train it
            create_and_train_model()
            model = load_trained_model()

        # Predict category
        category = predict_category(model, news_text)
        # Load target names
        newsgroups_train = fetch_20newsgroups(subset='train')
        category = newsgroups_train.target_names[category]

    return render(request, 'news/news_classification.html', {'category': category, 'news_text': news_text})
