import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb

import tensorflow as tf 
import tensorflow_hub as tf_hub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_score, confusion_matrix, roc_auc_score, roc_curve 

from utils import get_rows
from preprocess import preprocess

num_classes = 2
elmo_embedding_size = 1024
batch_size = 256

def generate_classifier_report(classifier,predictions,predictions_prob,name):
    print('Model stats for ' + name + ":")
    print('Accuracy:')
    print(accuracy_score(y_test,predictions))   
    print('Confusion matrix:')
    print(confusion_matrix(y_test,predictions).ravel())
    print('Mathews correlation coefficient:')
    print(matthews_corrcoef(y_test,predictions))
    print('Classification report:')
    print(classification_report(y_test,predictions))
    logit_roc_score = roc_auc_score(y_test,predictions)
    fpr, tpr, thresholds = roc_curve(y_test,predictions_prob)
    plt.figure()
    plt.plot(fpr,tpr,label=name + " AUC score:- " + str(logit_roc_score))
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate (FP / (FP + TN))')
    plt.ylabel('True Positive Rate (TP / (TP + FN))')
    plt.title('Receiver Operating Characteristics Curve')
    plt.legend(loc="lower right")
    plt.savefig(name + "_roc_curve")
    plt.show()
    
with tf.Session() as sess:    
    elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2",trainable=False)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    clf = RandomForestClassifier(warm_start=True)
    count = 0
    for dataset_chunk in pd.read_csv('../data/dataset/train.csv'):
        dataset = self.get_rows(dataset_chunk, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
        dataset_train, comment_seq_length, parent_comment_seq_length = preprocess(dataset, MAX_COMMENT_LENGTH, MAX_PARENT_COMMENT_LENGTH)
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
            average_reps = tf.math.reduce_mean(comment_embeddings, axis=2)
            average_reps_parent = tf.math.reduce_mean(parent_comment_embeddings, axis=2)
            clf.set_params(N_ESTIMATORS + (count * 100))
            clf.fit(comment_embeddings, dataset_train_batch['label'].to_list())
            count += 1