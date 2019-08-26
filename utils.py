import os
import numpy as np
import tensorflow as tf 

from preprocess import preprocess

def save_model(sess,path):
    saver = tf.train.Saver()
    saver.save(sess,path)

def load_model(sess,path):
    saver = tf.train.Saver()
    saver.restore(sess,path)    

def create_dictionary(dataset):
    for index,row in dataset.iterrows():
        tokens_comment = preprocess(str(row['parent_comment']) + " " + str(row['comment']))
        add_to_dictionary(tokens_comment)
    save_dictionary()

def populate_dictionary():
    if not os.path.isfile('data/processed/dictionary.txt'):
        starttime = time.time()
        create_dictionary(df_new)
        endtime = time.time()
        print("Time to create dictionary")
        print(endtime - starttime)
    else:   
        read_dictionary()
    print("Length of dictionary:- ")
    print(len(dictionary))

def save_dictionary():
    with open('data/processed/dictionary.txt','w') as file:
        file.writelines("%s\n" % word for word in dictionary)

def read_dictionary():
    with open('data/processed/dictionary.txt','r') as file:
        temp = file.read().splitlines()
        for i in range(0,len(temp)):
            dictionary.append(temp[i])
            
def add_to_dictionary(tokens):
    for token in tokens:
        if token not in dictionary:
            dictionary.append(token)
            
def save_state(deep_contextualized_embeddings_train, deep_contextualized_embeddings_parent_train, 
deep_contextualized_embeddings_test, deep_contextualized_embeddings_parent_test, X_train, X_test, y_train, y_test):
    np.save('data/trained_models/generated_embeddings_train.npy',deep_contextualized_embeddings_train)
    np.save('data/trained_models/generated_embeddings_parent_train.npy',deep_contextualized_embeddings_parent_train)
    np.save('data/trained_models/generated_embeddings_test.npy',deep_contextualized_embeddings_test)
    np.save('data/trained_models/generated_embeddings_parent_test.npy',deep_contextualized_embeddings_parent_test)
    np.save('data/trained_models/x_train.npy',X_train)
    np.save('data/trained_models/x_test.npy',X_test)
    np.save('data/trained_models/y_train.npy',y_train)
    np.save('data/trained_models/y_test.npy',y_test)