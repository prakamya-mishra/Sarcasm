import pandas as pd
import numpy as np
import os
import time
import seaborn as sb
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math

from bilstm import BiLSTM
from transformer import Transformer
from preprocess import get_max_length, get_max_length_parent, preprocess
from embeddings import get_deep_contextualized_embeddings
from utils import save_state

dictionary = []

embeddings = {}

df = pd.read_csv("data/dataset/train-balanced-sarcasm.csv")

df_new = df[['parent_comment','comment','label']]

sb.countplot(x='label',hue='label',data=df_new)

df_new = df_new.sample(10)

print(df_new.shape)

df_new.head()

for idx, row in df_new.iterrows():
    df_new.at[idx, 'comment'] = preprocess(row['comment'])
    df_new.at[idx, 'parent_comment'] = preprocess(row['parent_comment'])

#Remove nan here
X = df_new['comment']
y = df_new['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=222)

print(len(X_train))

print(len(y_train))

df_new_parent = df_new['parent_comment']
X_train_parent,X_test_parent,_,_ = train_test_split(df_new_parent,y,test_size=0.2,random_state=222)

print(len(X_train_parent))

print(len(X_test_parent))

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
word_embedding_size = 0 #For now as we are not using Glove
elmo_embedding_size = 1024
batch_size = 1000
epochs = 10
init_learning_rate = 0.01
decay_rate =  0.96
decay_steps = 8
dropout_rate = 0.1
feed_forward_op_size = 2048
num_attention_heads = 8 #Should divide embedding_size
num_encoder_blocks = 6

embedding_size = word_embedding_size + elmo_embedding_size

#Currenlty not using concatenation of Glove with ELMO
tf.reset_default_graph()
#Add support for transformer (Contextualized word embeddings with bidirectional self attention) here.
trans = Transformer(get_max_length(X_train), embedding_size, feed_forward_op_size, dropout_rate, num_encoder_blocks, num_attention_heads)
trans_parent = Transformer(get_max_length(X_train_parent), embedding_size, feed_forward_op_size, dropout_rate, num_encoder_blocks, num_attention_heads)
#Contextualized word embeddings with bidirectional self attention processed using LSTM.
bilstm = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
bilstm_parent = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
    softmax_w = tf.get_variable('W', shape=[2 * (elmo_embedding_size + word_embedding_size),1], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
    softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[1], dtype=tf.float32)

final_state = tf.concat([bilstm.final_state, bilstm_parent.final_state],0)
logit = tf.matmul(final_state,softmax_w) + softmax_b
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=bilstm.y)
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
            y_train_parent_batch = y_train[i * batch_size : min((i + 1) * batch_size,len(y_train))]
            sequence_lengths_parent_batch = sequence_lengths_parent_train[i * batch_size : min((i + 1) * batch_size,len(sequence_lengths_parent_train))]
            fetches = {
            'enc_input': trans.enc_input,
            'enc_input_parent': trans_parent.enc_input
            }
            feed_dict = {
            trans.x: X_train_batch,
            trans_parent.x: X_train_parent_batch
            }
            for i in X_train_batch.index:
                print(len(X_train_parent_batch[i]))
            resp = sess.run(fetches, feed_dict)
            fetches = {
                'cost': cost,
                'train_step': train_step,
                'global_step': global_step        
            }
            feed_dict = {
                bilstm.X : resp['enc_input'],
                bilstm.y : y_train_batch,
                bilstm.sequence_lengths : sequence_lengths_batch,
                bilstm_parent.X : resp['enc_input_parent'],
                bilstm_parent.y : y_train_parent_batch,
                bilstm_parent.sequence_lengths : sequence_lengths_parent_batch
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