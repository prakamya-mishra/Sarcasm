class BiLSTM():
    
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
