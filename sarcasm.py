import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import time
import math

from preprocess import preprocess
from utils import get_rows, sample_training_data
from bilstm import BiLSTM
from transformer import Transformer

COMMENT_LABEL = 'comment'
PARENT_COMMENT_LABEL = 'parent_comment'

DATASET_SIZE = 10000
CHUNKSIZE = 1000
TRAIN_SIZE = 0.8
MAX_COMMENT_LENGTH = 100
MAX_PARENT_COMMENT_LENGTH = 100

#Hyperparameters for the model
#Hyperparameter tuning required
num_classes = 2
word_embedding_size = 0 #For now as we are not using Glove
elmo_embedding_size = 1024
batch_size = 64
epochs = 10
init_learning_rate = 0.001
decay_rate =  0.96
decay_steps = 8
dropout_rate = 0.1
feed_forward_op_size = 2048
num_attention_heads = 8 #Should divide embedding_size
num_encoder_blocks = 6

embedding_size = word_embedding_size + elmo_embedding_size

dataset = pd.read_csv('data/dataset/train-balanced-sarcasm.csv')
dataset = dataset.sample(DATASET_SIZE)
print(dataset.head())

tf.reset_default_graph()
trans = Transformer(MAX_COMMENT_LENGTH, embedding_size, feed_forward_op_size, dropout_rate, num_encoder_blocks, num_attention_heads)
trans_parent = Transformer(MAX_PARENT_COMMENT_LENGTH, embedding_size, feed_forward_op_size, dropout_rate, num_encoder_blocks, num_attention_heads)
bilstm = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
bilstm_parent = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
    softmax_w = tf.get_variable('W', shape=[2 * feed_forward_op_size, num_classes], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
    softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[num_classes], dtype=tf.float32)
final_state = tf.concat([bilstm.final_state, bilstm_parent.final_state],1)
logit = tf.matmul(final_state,softmax_w) + softmax_b
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=bilstm.y, logits=logit)
cost = tf.reduce_mean(cost)
global_step = tf.Variable(0,name='global_step',trainable=False)
learning_rate = tf.train.exponential_decay(init_learning_rate,global_step,decay_steps, decay_rate,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(cost)
train_step = optimizer.apply_gradients(gradients,global_step=global_step)

with tf.Session() as sess:
    elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2",trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    for epoch in range(0,epochs):
        epoch_starttime = time.time()
        epoch_cost = 0
        chunk_id = 1
        processed_entries = 0
        for dataset_chunk in pd.read_csv('data/dataset/train-balanced-sarcasm.csv', chunksize=CHUNKSIZE):
            chunk_starttime = time.time()
            dataset = get_rows(dataset_chunk, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
            dataset_train, comment_seq_length, parent_comment_seq_length = preprocess(dataset, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
            dataset_train = sample_training_data(dataset_train, chunk_id)
            chunk_id += 1
            for i in range(0,int(math.floor(dataset_train.shape[0]/batch_size))):
                print('Low: ' + str(i * batch_size) + ' High: ' + str(min((i + 1) * batch_size, len(dataset_train.index))) + ' Size: ' + str(len(dataset_train.index)))
                dataset_train_batch = dataset_train.loc[dataset_train.index[i * batch_size: min((i + 1) * batch_size, len(dataset_train.index))]]
                comment_seq_length_batch = comment_seq_length[i * batch_size : min((i + 1) * batch_size,len(comment_seq_length))]
                parent_comment_seq_length_batch = parent_comment_seq_length[i * batch_size : min((i + 1) * batch_size,len(parent_comment_seq_length))]
                comment_embeddings_list = dataset_train_batch['comment'].to_list()
                parent_comment_embeddings_list = dataset_train_batch['parent_comment'].to_list()
                for j in range(0, len(comment_embeddings_list)):
                    comment_embeddings_list[j] = comment_embeddings_list[j].split()
                    parent_comment_embeddings_list[j] = parent_comment_embeddings_list[j].split()
                comment_embeddings = np.array(comment_embeddings_list)
                parent_comment_embeddings = np.array(parent_comment_embeddings_list)
                comment_embeddings = elmo(inputs={"tokens": comment_embeddings,"sequence_len": comment_seq_length_batch},signature='tokens',as_dict=True)["elmo"]
                parent_comment_embeddings = elmo(inputs={"tokens": parent_comment_embeddings,"sequence_len": parent_comment_seq_length_batch},signature='tokens',as_dict=True)["elmo"]
                comment_embeddings = sess.run(comment_embeddings)
                parent_comment_embeddings = sess.run(parent_comment_embeddings)
                comment_embeddings = np.array(comment_embeddings)
                parent_comment_embeddings = np.array(parent_comment_embeddings)
                fetches = {
                'enc_input': trans.enc_input,
                'enc_input_parent': trans_parent.enc_input
                }
                feed_dict = {
                trans.x: comment_embeddings,
                trans_parent.x: parent_comment_embeddings
                }
                resp = sess.run(fetches, feed_dict)
                fetches = {
                    'cost': cost,
                    'train_step': train_step,
                    'global_step': global_step        
                }
                feed_dict = {
                bilstm.X : resp['enc_input'],
                bilstm.y : dataset_train_batch['label'].to_list(),
                bilstm.sequence_lengths : comment_seq_length_batch,
                bilstm_parent.X : resp['enc_input_parent'],
                bilstm_parent.sequence_lengths : parent_comment_seq_length_batch
                }
                resp = sess.run(fetches,feed_dict)
                print('Global Step:- ')
                print(resp['global_step'])
                epoch_cost += resp['cost']
            chunk_endtime = time.time()
            print('Time takes to process chunk: ') 
            print(chunk_endtime - chunk_starttime)
            if processed_entries == DATASET_SIZE:
                break
            processed_entries += CHUNKSIZE
        epoch_endtime = time.time()
        print('Time takes for epoch ' + str(epoch) + ': ')
        print(epoch_endtime - epoch_starttime)
        print('Epoch cost: ')
        print(epoch_cost)