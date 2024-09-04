import io
import random
import string
import warnings
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk

# Download the 'punkt' package
nltk.download('punkt')


# import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)  # for downloading packages

# Lemmatization
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Load the chatbot text file
def load_chatbot_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None
    
    with open(filepath, 'r', encoding='utf8', errors='ignore') as fin:
        return fin.read().lower()

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad you are talking to me!", "Hello, how can I assist you today?"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def response(user_response, sent_tokens):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.pop()  # Remove user_response from the tokens
    return robo_response

def chatbot():
    # Load data
    raw = load_chatbot_data('chatbot.txt')
    if raw is None:
        return

    sent_tokens = nltk.sent_tokenize(raw)
    word_tokens = nltk.word_tokenize(raw)

    # Start interaction
    print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

    while True:
        user_response = input().strip().lower()
        
        if user_response in ('bye', 'exit', 'quit'):
            print("ROBO: Bye! Take care.")
            break
        
        elif user_response in ('thanks', 'thank you'):
            print("ROBO: You're welcome!")
            break
        
        else:
            if greeting(user_response):
                print(f"ROBO: {greeting(user_response)}")
            else:
                print(f"ROBO: {response(user_response, sent_tokens)}")

if __name__ == "__main__":
    chatbot()
