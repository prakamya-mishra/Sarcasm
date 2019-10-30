import tensorflow as tf

class Models:
    
    def __init__(self, max_length, embed_size, elmo_embed_size):
        self.X = tf.placeholder(shape=[None,None,embed_size + elmo_embed_size],dtype=tf.float32,name='X')
        self.X_parent = tf.placeholder(shape=[None, None, embed_size + elmo_embed_size], dtype=tf.float32, name='X_parent')
        self.y = tf.placeholder(shape=[None],dtype=tf.int64,name='y')
        self.sequence_lengths = tf.placeholder(shape=[None],dtype=tf.int32,name='sequence_lengths')
        self.max_length = max_length
        self.elmo_embed_size = elmo_embed_size
        self.embed_size = embed_size
        self.hidden_size = elmo_embed_size + embed_size
        
    def convolution(self, filter_sizes, num_filters):
        assert len(filter_sizes) == len(num_filters)
        with tf.variable_scope('conv_2d', reuse=tf.AUTO_REUSE):
            pooled_outputs = []
            for i in range(0, len(filter_sizes)):
                filter_shape = [filter_sizes[i], self.hidden_size, 1, num_filters[i]]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name="b")
                conv = tf.nn.conv2d(self.X, W, strides=[1, 1, 1, 1], padding='VALID', name='conv') + b
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(relu, ksize=[1,self.max_length – filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_outputs.append(pooled)
    
    def linear_combination(self):
        with tf.variable_scope('linear_combination', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[self.max_length], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            output_x = tf.matmul(self.X, w)
            output_x_parent = tf.matmul(self.X_parent, w)
            return tf.concat([output_x, output_x_parent], 1)
             