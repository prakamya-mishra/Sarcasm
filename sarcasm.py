import pandas as pd
import numpy as np
from numpy import nan
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_hub as tf_hub
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import math
import sys
import getopt
import shutil

from preprocess import preprocess
from bilstm import BiLSTM
from transformer import Transformer
from config import SENDGRID_API_KEY
#from models import Models

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

import firebase_admin
from firebase_admin import credentials, db

gcp_credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=gcp_credentials)

cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred, {
'databaseURL': 'https://original-dryad-251711.firebaseio.com/'
})

BASE_REF = '/'
COMMENT_LABEL = 'comment'
PARENT_COMMENT_LABEL = 'parent_comment'

DATASET_SIZE = 30000
CHUNKSIZE = 5000
TRAIN_SIZE = 0.8
MAX_COMMENT_LENGTH = 100
MAX_PARENT_COMMENT_LENGTH = 100
RANDOM_SEED = 222
MODEL_CHECKPOINT_DURATION = 1

#Hyperparameters for the model
#Hyperparameter tuning required
num_classes = 2
intermediate_layer_size_1 = 1024
intermediate_layer_size_2 = 512
intermediate_layer_size_3 = 128
lambda_l2_reg = 0.005
word_embedding_size = 0 #For now as we are not using Glove
elmo_embedding_size = 1024
batch_size = 256
epochs = 5
init_learning_rate = 0.001
decay_rate =  0.96
decay_steps = 8
dropout_rate = 0.1
feed_forward_op_size = 2048
num_attention_heads = 8 #Should divide embedding_size
num_encoder_blocks = 6

embedding_size = word_embedding_size + elmo_embedding_size

base_firebase_ref = db.reference(BASE_REF)
    
class TrainModel:
    
    def __init__(self, dataset_path, debug):
        self.dataset_path = dataset_path
        self.debug = debug
        self.build_model()
        
    def build_model(self):
        self.bilstm = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
        self.bilstm_parent = BiLSTM(num_classes,word_embedding_size,elmo_embedding_size,batch_size,epochs,init_learning_rate,decay_rate,decay_steps)
        with tf.variable_scope('softmax-1',reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable('W', shape=[2 * feed_forward_op_size, intermediate_layer_size_1], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[intermediate_layer_size_1], dtype=tf.float32)    
        self.final_state = tf.concat([self.bilstm.final_state, self.bilstm_parent.final_state],1)
        self.logit = tf.matmul(self.final_state,softmax_w) + softmax_b
        self.logit = tf.nn.relu(self.logit)
        with tf.variable_scope('softmax-2', reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable('W', shape=[intermediate_layer_size_1, intermediate_layer_size_2], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[intermediate_layer_size_2], dtype=tf.float32)
        self.logit = tf.matmul(self.logit,softmax_w) + softmax_b       
        self.logit = tf.nn.relu(self.logit) 
        with tf.variable_scope('softmax-3', reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable('W', shape=[intermediate_layer_size_2, intermediate_layer_size_3], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[intermediate_layer_size_3], dtype=tf.float32)    
        self.logit = tf.matmul(self.logit,softmax_w) + softmax_b    
        self.logit = tf.nn.relu(self.logit)
        with tf.variable_scope('softmax-4', reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable('W', shape=[intermediate_layer_size_3, num_classes], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            softmax_b = tf.get_variable('b',initializer=tf.constant_initializer(0.0), shape=[num_classes], dtype=tf.float32)  
        self.logit = tf.matmul(self.logit,softmax_w) + softmax_b              
        self.norm_logit = tf.nn.softmax(self.logit)
        self.predictions = tf.cast(tf.math.argmax(self.norm_logit, axis=1), tf.int64)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.bilstm.y), tf.float32))
        
    def get_rows(self, dataset, max_comment_length, max_parent_comment_length):
        comments = []
        parent_comments = []
        labels = []
        for index, row in dataset.iterrows():
            if(isinstance(row[COMMENT_LABEL], str) and isinstance(row[PARENT_COMMENT_LABEL], str)):
                if(len(row[COMMENT_LABEL]) <= max_comment_length and len(row[PARENT_COMMENT_LABEL]) <= max_parent_comment_length 
                    and row[COMMENT_LABEL] is not nan and row[PARENT_COMMENT_LABEL] is not nan 
                    and len(row[COMMENT_LABEL]) > 0 and len(row[PARENT_COMMENT_LABEL]) > 0):
                    comments.append(row[COMMENT_LABEL])
                    parent_comments.append(row[PARENT_COMMENT_LABEL])
                    labels.append(row['label'])
        return pd.DataFrame({COMMENT_LABEL: comments, PARENT_COMMENT_LABEL: parent_comments, 'label': labels})

    def sample_training_data(self, dataset, batch_id, training=True):
        np.random.seed(RANDOM_SEED)
        mask = np.random.rand(dataset.shape[0]) < TRAIN_SIZE
        #dataset[~mask].to_csv('../data/test/batch_' + str(batch_id) + ".csv")
        return dataset[mask] if training else dataset[~mask]   
        
    def train(self, notify_progress, pretrained_model_path=None):
        try:
            self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.bilstm.y, logits=self.logit)
            self.l2 = lambda_l2_reg * tf.reduce_sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not "b" in tf_var.name])
            self.cost = tf.reduce_mean(self.cost) + self.l2
            self.global_step = tf.Variable(0,name='global_step',trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
            self.gradients = self.optimizer.compute_gradients(self.cost)
            self.train_step = self.optimizer.apply_gradients(self.gradients,global_step=self.global_step)
            config = tf.ConfigProto(device_count={'CPU': 8})
            with tf.Session(config=config) as sess:
                if self.debug:
                    writer = tf.summary.FileWriter('../data/graphs', sess.graph)
                if not self.debug:
                    base_firebase_ref.delete()
                elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2",trainable=False)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                if (pretrained_model_path is not None) :
                    saver = tf.train.Saver()
                    saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_path))
                    log(tf.train.latest_checkpoint(pretrained_model_path), self.debug)
                for epoch in range(1,epochs + 1):
                    epoch_starttime = time.time()
                    epoch_cost = 0
                    chunk_id = 1
                    global_step_count = 0
                    processed_entries = 0
                    epoch_predictions = []
                    epoch_expected = []
                    for dataset_chunk in pd.read_csv(self.dataset_path, chunksize=CHUNKSIZE):
                        processed_entries += CHUNKSIZE
                        if processed_entries > DATASET_SIZE:
                            break
                        chunk_starttime = time.time()
                        dataset = self.get_rows(dataset_chunk, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
                        dataset_train, comment_seq_length, parent_comment_seq_length = preprocess(dataset, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
                        #dataset_train = self.sample_training_data(dataset_train, chunk_id)
                        chunk_id += 1
                        for i in range(0,int(math.floor(dataset_train.shape[0]/batch_size))):
                            log('Low: ' + str(i * batch_size) + ' High: ' + str(min((i + 1) * batch_size, len(dataset_train.index))) + ' Size: ' + str(len(dataset_train.index)), self.debug)
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
                            comment_embeddings = sess.run(comment_embeddings, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                            parent_comment_embeddings = sess.run(parent_comment_embeddings, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                            comment_embeddings = np.array(comment_embeddings)
                            parent_comment_embeddings = np.array(parent_comment_embeddings)
                            fetches = {
                                'cost': self.cost,
                                'train_step': self.train_step,
                                'global_step': self.global_step,
                                'predictions':self.predictions        
                            }
                            feed_dict = {
                            self.bilstm.X : comment_embeddings,
                            self.bilstm.y : dataset_train_batch['label'].to_list(),
                            self.bilstm.sequence_lengths : comment_seq_length_batch,
                            self.bilstm_parent.X : parent_comment_embeddings,
                            self.bilstm_parent.sequence_lengths : parent_comment_seq_length_batch
                            }
                            resp = sess.run(fetches,feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                            global_step_count = resp['global_step']
                            epoch_cost += resp['cost']
                            epoch_predictions.extend(resp['predictions'])
                            epoch_expected.extend(dataset_train_batch['label'].to_list())
                        chunk_endtime = time.time()
                        log('Time takes to process chunk: ', self.debug) 
                        log(str(chunk_endtime - chunk_starttime), self.debug)
                        log('Current global step: ', self.debug)
                        log(str(global_step_count), self.debug)
                    epoch_endtime = time.time()
                    log('Testing model: ', self.debug)
                    test_accuracy_score, test_accuracy_tf, test_confusion_matrix, test_time = self.test(sess, elmo, '../data/test/test.csv')
                    epoch_summary = 'Epoch Summary [' + str(time.time()) + '] :\nTime taken for epoch ' 
                    epoch_summary += str(epoch) + ': \n' + str(epoch_endtime - epoch_starttime) + '\n'
                    epoch_summary += 'Epoch cost: \n' + str(epoch_cost) + '\nTest Results (train): \n'
                    epoch_summary += 'Accuracy Score: \n' + str(accuracy_score(epoch_expected, epoch_predictions)) + '\nConfusion Matrix: ' + str(confusion_matrix(epoch_expected, epoch_predictions))
                    epoch_summary += '\nTime taken to test: \n' + str(test_time) + '\n'
                    epoch_summary += 'Test results: \n' + 'Accuracy (tf): ' + str(test_accuracy_tf)
                    epoch_summary += '\nAccuracy (accuracy_score): ' + str(test_accuracy_score)  + '\nConfusion Matrix: ' + str(test_confusion_matrix)
                    log(epoch_summary, self.debug)
                    if notify_progress:
                        send_mail('Model training progress', '<strong>' + epoch_summary +  '</strong>', self.debug)
                    if(epoch % MODEL_CHECKPOINT_DURATION == 0 or epoch == epochs):
                        saver = tf.train.Saver()
                        if os.path.isdir('../data/trained_models_gcp/checkpoint_3_' + str(epoch)):
                            shutil.rmtree('../data/trained_models_gcp/checkpoint_3_' + str(epoch))
                        os.mkdir('../data/trained_models_gcp/checkpoint_3_' + str(epoch))    
                        saver.save(sess, '../data/trained_models_gcp/checkpoint_3_' + str(epoch) + '/model', global_step=self.global_step)
        except Exception as exception:
            log(str(exception), self.debug)
            if not self.debug:
                send_mail('Training failed', '<strong>Model training failed. Check logs.</strong><p>' + str(exception) + '</p>', self.debug)
                request = service.instances().stop(project='majestic-disk-257314', zone='us-central1-a', instance='7273726640686567037')
                response = request.execute()
                log(response, self.debug)
                
    def test(self, sess, elmo, test_dataset_path, trained_model_path=None):
        predictions = []
        accuracy = []
        test_accuracy_score = 0
        try:
            test_dataset = pd.read_csv(test_dataset_path)
            test_dataset = self.get_rows(test_dataset, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
            test_dataset, comment_seq_length, parent_comment_seq_length = preprocess(test_dataset, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
            if (trained_model_path is not None):
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(trained_model_path))
            starttime = time.time()    
            for i in range(0,int(math.floor(test_dataset.shape[0]/batch_size))):
                log('Low: ' + str(i * batch_size) + ' High: ' + str(min((i + 1) * batch_size, len(test_dataset.index))) + ' Size: ' + str(len(test_dataset.index)), self.debug)
                dataset_test_batch = test_dataset.loc[test_dataset.index[i * batch_size: min((i + 1) * batch_size, len(test_dataset.index))]]
                comment_seq_length_batch = comment_seq_length[i * batch_size : min((i + 1) * batch_size,len(comment_seq_length))]
                parent_comment_seq_length_batch = parent_comment_seq_length[i * batch_size : min((i + 1) * batch_size,len(parent_comment_seq_length))]
                comment_embeddings_list = dataset_test_batch['comment'].to_list()
                parent_comment_embeddings_list = dataset_test_batch['parent_comment'].to_list()
                for j in range(0, len(comment_embeddings_list)):
                    comment_embeddings_list[j] = comment_embeddings_list[j].split()
                    parent_comment_embeddings_list[j] = parent_comment_embeddings_list[j].split()
                comment_embeddings = np.array(comment_embeddings_list)
                parent_comment_embeddings = np.array(parent_comment_embeddings_list)
                comment_embeddings = elmo(inputs={"tokens": comment_embeddings,"sequence_len": comment_seq_length_batch},signature='tokens',as_dict=True)["elmo"]
                parent_comment_embeddings = elmo(inputs={"tokens": parent_comment_embeddings,"sequence_len": parent_comment_seq_length_batch},signature='tokens',as_dict=True)["elmo"]
                comment_embeddings = sess.run(comment_embeddings, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                parent_comment_embeddings = sess.run(parent_comment_embeddings, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                comment_embeddings = np.array(comment_embeddings)
                parent_comment_embeddings = np.array(parent_comment_embeddings)
                fetches = {
                    'logit': self.logit,
                    'norm_logit': self.norm_logit,
                    'predictions': self.predictions,
                    'accuracy': self.accuracy       
                }
                feed_dict = {
                self.bilstm.X : comment_embeddings,
                self.bilstm.y : dataset_test_batch['label'].to_list(),
                self.bilstm.sequence_lengths : comment_seq_length_batch,
                self.bilstm_parent.X : parent_comment_embeddings,
                self.bilstm_parent.sequence_lengths : parent_comment_seq_length_batch
                }
                resp = sess.run(fetches,feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                predictions.extend(resp['predictions'])
                accuracy.append(resp['accuracy'])
            endtime = time.time()
            log('Time taken to test: ', self.debug)
            log(str(endtime - starttime), self.debug)    
            log('Model accuracy (tf): ' + str(accuracy), self.debug)            
            test_accuracy_score = accuracy_score(test_dataset['label'].to_list()[0:len(predictions)], predictions)
            log('Model accuracy (accuracy_score): ' + str(test_accuracy_score), self.debug)
        except Exception as exception:
            log(str(exception), self.debug)
            raise exception
        return test_accuracy_score, accuracy, confusion_matrix(test_dataset['label'].to_list()[0:len(predictions)], predictions),(endtime - starttime)
            
def send_mail(subject, html_content, debug):
    message = Mail(
    from_email='sarcasm-vm-instance@gcp.com',
    to_emails='sk261@snu.edu.in',
    subject=subject,
    html_content=html_content
    )
    try:
        send_grid = SendGridAPIClient(SENDGRID_API_KEY)
        response = send_grid.send(message)
        log(str(response.status_code), debug)
        log(response.body, debug)
    except Exception as exception:
        log(str(exception), debug)        
            
def train(debug, notify_progress):
    tf.reset_default_graph()
    model = TrainModel('data/dataset/train-balanced-sarcasm.csv', debug)
    model.train(notify_progress, '../data/trained_models_gcp/checkpoint_3')
    
def test(debug):
    tf.reset_default_graph()
    model = TrainModel('data/dataset/train-balanced-sarcasm.csv', debug)
    with tf.Session() as sess:
        elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2",trainable=False)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        model.test(sess, elmo, '../data/test/test.csv', '../data/trained_models_gcp/checkpoint_2')
    
def log(message, debug):
    if debug:
        print(message)
    else:
        base_firebase_ref.push().set(message)
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        debug = False
        notify_progress = False
        opts, args = getopt.getopt(argv, 'dp')
        for opt in opts:
            if opt[0] == '-d':
                debug = True
            elif opt[0] == '-p':
                notify_progress = True
        if not os.path.isdir('../data'):
            os.mkdir('../data')
            os.mkdir('../data/test')
            os.mkdir('../data/trained_models')   
        train(debug, notify_progress)
    except getopt.GetoptError as exception:
        print(exception)
        sys.exit(2)
