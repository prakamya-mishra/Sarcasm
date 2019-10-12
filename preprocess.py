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

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    for i in range(0,len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i],pos=str(get_pos_tag(tokens[i])))
    return tokens

def preprocess_sentence(sentence):
    processed_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    processed_sentence = processed_sentence.lower().strip()
    tokens_comment = word_tokenize(processed_sentence)
    tokens_comment = remove_stopwords(tokens_comment)
    tokens_comment = lemmatize(tokens_comment)
    return ' '.join(tokens_comment)
    
def pad_sentences(sentence, max_length):
   new_sentence = sentence.split()
   for i in range(len(new_sentence), max_length):
       new_sentence.append('<PAD>')
   return ' '.join(new_sentence)

def preprocess(dataset, max_comment_length, max_parent_comment_length):
    count = 0
    count_parent = 0
    comment_seq_length = []
    parent_comment_seq_length = []
    for idx, row in dataset.iterrows():
        preprocessed_comment = preprocess_sentence(row['comment'])
        preprocessed_parent_comment = preprocess_sentence(row['parent_comment'])
        if(isinstance(preprocessed_comment, str) and isinstance(preprocessed_parent_comment, str)):
            comment_seq_length.append(len(preprocessed_comment))
            parent_comment_seq_length.append(len(preprocessed_parent_comment))
            dataset.at[idx, 'comment'] = pad_sentences(preprocessed_comment, max_comment_length)
            dataset.at[idx, 'parent_comment'] = pad_sentences(preprocessed_parent_comment, max_parent_comment_length)
            if len(row['comment']) == max_comment_length:
                count += 1
            if len(row['parent_comment']) == max_parent_comment_length:
                count_parent += 1
    if count == count_parent and count == dataset.shape[0]:
        print('Data preprocessing successfull')
    return dataset, comment_seq_length, parent_comment_seq_length
    
def build_subj_lex_dict(subj_lex_file_path):
    subj_lex_dict = {}
    with open(subj_lex_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split()
            word = tokens[2][6:]
            pos = tokens[3][5:]
            polarity = tokens[5][14:]
            type = tokens[0][5:]
            word_pos_dict = {pos: {
            'polarity': polarity,
            'type': type
            }}
            if word in subj_lex_dict:
                subj_lex_dict[word].append(word_pos_dict)
            else:
                subj_lex_dict[word] = [word_pos_dict]
    return subj_lex_dict