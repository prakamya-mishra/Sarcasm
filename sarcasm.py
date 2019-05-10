import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
import os
import time
import seaborn as sb
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from numpy import nan
import math

class LSTM():
    
    def __init__(self,num_classes,elmo_embed_size,embed_size,batch_size,epochs,init_learning_rate,decay_steps,decay_rate):
        self.X = tf.placeholder(shape=[None,None,embed_size + elmo_embed_size],dtype=tf.float32,name='X')
        self.y = tf.placeholder(shape=[None],dtype=tf.int64,name='y')
        self.sequence_lengths = tf.placeholder(shape=[None],dtype=tf.int32,name='sequence_lengths')
        self.num_classes = num_classes
        self.elmo_embed_size = elmo_embed_size
        self.embed_size = embed_size
        self.hidden_size = elmo_embed_size + embed_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.deacy_rate = decay_rate
        self.model()
    
    def model(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope('Bi-Directional-LSTM',reuse=tf.AUTO_REUSE):
            output_vals,output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell_fw,
            cell_bw = cell_bw,
            inputs = self.X,
            sequence_length = self.sequence_lengths,
            dtype = tf.float32)
        self.final_state = tf.concat([output_states[0].c,output_states[1].c],axis=1)

            save_model(sess,path)
            
    def test(self,X_test,y_test,sequence_lengths_test,path):
        with tf.Session() as sess:
            starttime = time.time()
            load_model(sess,path)
            fetches = {
                'accuracy': self.accuracy,
                'predictions': self.predictions
            }
            feed_dict = {
                self.X : X_test,
                self.y : y_test,
                self.sequence_lengths: sequence_lengths_test
            }
            resp = sess.run(fetches,feed_dict)
            endtime = time.time()
            print('Time to test model:- ')
            print(endtime - starttime)
            print('Model accuracy:- ')
            print(resp['accuracy'])
            print('Model predictions:- ')
            print(resp['predictions'])

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

def add_to_dictionary(tokens):
    for token in tokens:
        if token not in dictionary:
            dictionary.append(token)

def save_dictionary():
    with open('data/processed/dictionary.txt','w') as file:
        file.writelines("%s\n" % word for word in dictionary)

def read_dictionary():
    with open('data/processed/dictionary.txt','r') as file:
        temp = file.read().splitlines()
        for i in range(0,len(temp)):
            dictionary.append(temp[i])

def save_model(sess,path):
    saver = tf.train.Saver()
    saver.save(sess,path)

def load_model(sess,path):
    saver = tf.train.Saver()
    saver.restore(sess,path)

def preprocess(sentence):
    processed_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    tokens_comment = word_tokenize(processed_sentence)
    tokens_comment = remove_stopwords(tokens_comment)
    tokens_comment = lemmatize(tokens_comment)
    return tokens_comment

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

def pad_tokens(tokens,max_length):
    zeros = np.zeros(len(tokens[0]))
    while len(tokens) < max_length:
        tokens = np.vstack([tokens,zeros])
    return tokens

def get_max_length(X):
    max_length = 0
    index = 0
    for i in range(0,len(X.index)):
        if(X[i:i+1][X.index[i]] is not nan):
            preprocessed_tokens = preprocess(X[i:i+1][X.index[i]])
            if(len(preprocessed_tokens) < 30):
                if max_length < len(preprocessed_tokens):
                    max_length = len(preprocessed_tokens)
                    index = i
    print(index)
    print(preprocess(X[index:index+1][X.index[index]]))
    return max_length

def get_max_length_parent(X_train,index):
    max_length = 0
    ind = 0
    for i in range(0,len(index)):
        processed_tokens = preprocess(X_train[index[i]])
        if max_length < len(processed_tokens):
            max_length = len(processed_tokens)
            ind = i
    print(ind)
    print(preprocess(X_train[index[ind]]))
    return max_length

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

def save_state():
    np.save('data/trained_models/generated_embeddings_train.npy',deep_contextualized_embeddings_train)
    np.save('data/trained_models/generated_embeddings_parent_train.npy',deep_contextualized_embeddings_parent_train)
    np.save('data/trained_models/generated_embeddings_test.npy',deep_contextualized_embeddings_test)
    np.save('data/trained_models/generated_embeddings_parent_test.npy',deep_contextualized_embeddings_parent_test)
    np.save('data/trained_models/x_train.npy',X_train)
    np.save('data/trained_models/x_test.npy',X_test)
    np.save('data/trained_models/y_train.npy',y_train)
    np.save('data/trained_models/y_test.npy',y_test)

stop_words = set(stopwords.words('english'))

dictionary = []

embeddings = {}

elmo = hub.Module("https://tfhub.dev/google/elmo/2",trainable=True)

df = pd.read_csv("data/dataset/train-balanced-sarcasm.csv")

df_new = df[['parent_comment','comment','label']]

sb.countplot(x='label',hue='label',data=df_new)

df_new = df_new.sample(20000)

df_new.shape

df_new.head()

#Remove nan here
X = df_new['comment']
y = df_new['label']
X.reset_index()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=222)

X_train.shape

X_train.index

X_test.shape

X_test.index

df_new_parent = df_new['parent_comment']
X_train_parent,X_test_parent,_,_ = train_test_split(df_new_parent,y,test_size=0.2,random_state=222)

X_train_parent.shape

X_train_parent.index

X_test_parent.shape

X_test_parent.index

deep_contextualized_embeddings_train,y_pred_train,sequence_lengths_train = get_deep_contextualized_embeddings(X_train,y_train,get_max_length(X_train))

deep_contextualized_embeddings_parent_train,_,sequence_lengths_parent_train = get_deep_contextualized_embeddings(X_train_parent,y_train,get_max_length_parent(X_train_parent,X_train.index))

deep_contextualized_embeddings_test,y_pred_test,sequence_lengths_test = get_deep_contextualized_embeddings(X_test,y_test,get_max_length(X_test))

deep_contextualized_embeddings_parent_test,_,sequence_lengths_parent_test = get_deep_contextualized_embeddings(X_test_parent,y_test,get_max_length(X_test_parent))

deep_contextualized_embeddings_train = np.array(deep_contextualized_embeddings_train)
deep_contextualized_embeddings_parent_train = np.array(deep_contextualized_embeddings_parent_train)
deep_contextualized_embeddings_test = np.array(deep_contextualized_embeddings_test)
deep_contextualized_embeddings_parent_test = np.array(deep_contextualized_embeddings_parent_test)

deep_contextualized_embeddings_train.shape

deep_contextualized_embeddings_parent_train.shape

deep_contextualized_embeddings_test.shape

deep_contextualized_embeddings_parent_test.shape

#Hyperparameters for the model
#Hyperparameter tuning required
num_classes = 2
word_embedding_size = 0 #For now as we are currently not using Glove
elmo_embedding_size = 1024
batch_size = 1000
epochs = 10 #To be increased as the size of the dataset increases(current size being considered:- 10000 data points (8000 - Train,2000 - Test))
init_learning_rate = 0.01 #To be changed to exponentially decreasing value based on epochs passed
decay_rate =  0.96
decay_steps = 8

#Currenlty not using concatenation of Glove with ELMO
tf.reset_default_graph()
lstm = LSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
lstm_parent = LSTM(num_classes,word_embeddings_size,elmo_embedding_size,batch_size,epochs,init_learning,decay_rate,decay_steps)
with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
    softmax_w = tf.get_variable('W',initializer=tf.truncated_normal_initializer(shape=[4 * hidden_size,1]),dtype=tf.float32)
    softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0,shape=[1]),dtype=tf.float32)
final_state = tf.concat([lstm.final_state,lstm_parent.final_state],0)
logit = tf.matmul(final_state,softmax_w) + softmax_b
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=lstm.y)
global_step = tf.Variable(0,name='global_step',trainable=False)
learning_rate = tf.train.exponential_decay(init_learning_rate,global_step,decay_steps,
                                                decay_rate,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(cost)
train_step = optimizer.apply_gradients(gradients,global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(0,epochs):
        starttime = time.time()
        epoch_cost = 0
        for i in range(0,math.ceil(X_train.shape[0]/batch_size)):
            X_train_batch = X_train[i * batch_size : min((i + 1) * batch_size,X_train.shape[0])]
            y_train_batch = y_train[i * batch_size : min((i + 1) * batch_size,len(y_train))]
            sequence_lengths_batch = sequence_lengths_train[i * batch_size : min((i + 1) * batch_size,len(sequence_lengths_train))]
            X_train_parent_batch = X_train_parent[i * batch_size : min((i + 1) * batch_size,X_train.shape[0])]
            y_train_parent_batch = y_train_parent[i * batch_size : min((i + 1) * batch_size),X_train.shape[0]]
            sequence_lengths_parent_batch = sequence_lengths_parent_train[i * batch_size : min((i + 1) * batch_size,len(sequence_lengths_parent_train))]
            fetches = {
                'cost': cost,
                'train_step': train_step,
                'global_step': global_step        
            }
            feed_dict = {
                lstm.X : X_train_batch,
                lstm.y : y_train_batch,
                lstm.sequence_lengths : sequence_lengths_batch,
                lstm_parent.X : X_train_parent_batch,
                lstm_parent.y : y_train_parent_batch,
                lstm_parent.sequence_lengths : sequence_lengths_parent_batch
            }
            resp = sess.run(fetches,feed_dict)
            print('Global Step:- ')
            print(resp['global_step'])
            epoch_cost += resp['cost']
        endtime = time.time()
        print('Time to train epoch ' + str(j) + ':-')
        print(endtime - starttime)
        print('Epoch ' + str(j) + " cost :-")
        print(epoch_cost)
    save_state()

