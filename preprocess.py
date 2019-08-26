import re
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from numpy import nan

stop_words = set(stopwords.words('english'))

def get_pos_tag(token):
    pos_tag = nltk.pos_tag([token])[0][1]
    if pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_stopwords(tokens):
    tokens_wo_stopwords = []
    for i in range(0,len(tokens)):
        if tokens[i].lower() not in stop_words:
            tokens_wo_stopwords.append(tokens[i].lower())
    return tokens_wo_stopwords

def remove_stopwords(tokens):
    tokens_wo_stopwords = []
    for i in range(0,len(tokens)):
        if tokens[i].lower() not in stop_words:
            tokens_wo_stopwords.append(tokens[i].lower())
    return tokens_wo_stopwords

def get_pos_tag(token):
    pos_tag = nltk.pos_tag([token])[0][1]
    if pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    for i in range(0,len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i],pos=str(get_pos_tag(tokens[i])))
    return tokens


def preprocess(sentence):
    if sentence is not nan and len(sentence) > 0:
        processed_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        tokens_comment = word_tokenize(processed_sentence)
        tokens_comment = remove_stopwords(tokens_comment)
        tokens_comment = lemmatize(tokens_comment)
        return tokens_comment
    else:
        return "<PAD>"

def pad_tokens(tokens,max_length):
    zeros = np.zeros(len(tokens[0]))
    while len(tokens) < max_length:
        tokens = np.vstack([tokens,zeros])
    return tokens

def get_max_length(X):
    max_length = 0
    index = 0
    for i in X.index:
        if max_length < len(X[i]):
            max_length = len(X[i])
            index = i
    return max_length
    
def get_max_length_parent(X_train,index):
    max_length = 0
    ind = 0
    for i in range(0,len(index)):
        if max_length < len(X_train[index[i]]):
            max_length = len(X_train[index[i]])
            ind = i
    return max_length
