import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from preprocess import preprocess

elmo = hub.Module("https://tfhub.dev/google/elmo/2",trainable=True)

def populate_embeddings_dict():
    starttime = time.time()
    with open('data/processed/glove.6B.300d.txt','r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:])
            embeddings[word] = word_embedding
    endtime = time.time()
    print("Time taken to load embeddings:- ")
    print(endtime - starttime)

def embedding_lookup(x,embedding_dim=300):
    if(len(embeddings) == 0):
        populate_embeddings_dict()
    embedding = []
    for i in range(0,len(x)):
        if(x[i] in embeddings):
            embedding.append(embeddings[x[i]])
        else:
            zero_arr = np.zeros(embedding_dim).tolist()
            embedding.append(zero_arr)
    return np.array(embedding)

def get_elmo_embeddings(sess,tokens_input,tokens_length):
    embeddings = elmo(inputs={"tokens": tokens_input,"sequence_len": tokens_length},signature='tokens',as_dict=True)["elmo"]
    return sess.run(embeddings)

def get_deep_contextualized_embeddings(X,y,max_length):
    deep_contextualized_embeddings = []
    sequence_lengths = []
    elmo_tokens = []
    elmo_tokens_length = []
    elmo_embeddings_list = []
    y_pred = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        starttime = time.time()
        for i in range(0,len(X.index)):
            if(X[i:i+1][X.index[i]] is not nan):
                preprocessed_tokens = preprocess(X[i:i+1][X.index[i]])
                if(len(preprocessed_tokens) < max_length + 1):
                    sequence_lengths.append(len(preprocessed_tokens))
                    y_pred.append(y[i:i+1][y.index[i]])
                    for j in range(len(preprocessed_tokens),max_length):
                        preprocessed_tokens.append("<PAD>")
                    #word_embedding = embedding_lookup(preprocessed_tokens)
                    #word_embedding = np.array(pad_tokens(word_embedding,max_length))
                    elmo_tokens.append(preprocessed_tokens)
                    elmo_tokens_length.append(len(preprocessed_tokens))
                    #deep_contextualized_embeddings.append(np.hstack([word_embedding,elmo_embedding]))
                    if (i + 1) % 1000 == 0:
                        elmo_embedding = get_elmo_embeddings(sess,np.array(elmo_tokens),np.array(elmo_tokens_length))
                        for j in range(0,len(elmo_embedding)):
                            deep_contextualized_embeddings.append(np.array(pad_tokens(elmo_embedding[j],max_length)))
                        temp_arr = np.array(deep_contextualized_embeddings)
                        print(temp_arr.shape)
                        elmo_tokens.clear()
                        elmo_tokens_length.clear()
        endtime = time.time()
        print("Total time to generate embeddings:- ")
        print(endtime - starttime)
    return np.array(deep_contextualized_embeddings),np.array(y_pred),np.array(sequence_lengths)