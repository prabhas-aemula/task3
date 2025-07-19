import random
import json
import pickle
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as f:
    intents = json.load(f)

# Preprocessing
corpus = []
labels = []
all_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        corpus.append(' '.join(tokens))
        labels.append(intent['tag'])
    all_tags.append(intent['tag'])

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y = labels

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Predict function
def chatbot_response(user_input):
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    cleaned = ' '.join(tokens)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    for intent in intents['intents']:
        if intent['tag'] == pred:
            return random.choice(intent['responses'])

# Run Chatbot
print("🤖 Chatbot is ready! Type 'quit' to exit.")
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        print("Bot: Goodbye!")
        break
    reply = chatbot_response(inp)
    print("Bot:", reply)
