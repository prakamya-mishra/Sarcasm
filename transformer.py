import numpy as np
import tensorflow as tf

class Transformer:
    
    def __init__(self, embedding_size, max_sen_len, dim_k, dim_q, dim_feed_for, dim_model, dropout_rate, num_enc_blocks, num_att_heads):
        self.x = tf.placeholder(shape=[None, max_sent_len, embedding_size])
        self.embedding_size = embedding_size
        self.max_sent_len = max_sent_len
        self.dim_k = dim_k
        self.dim_q = dim_q
        self.dim_feed_for = dim_feed_for
        self.dim_model = self.dim_model
        self.dropout_rate = dropout_rate
        self.num_enc_blocks = num_enc_blocks
        self.num_att_heads = num_att_heads
        
    def encode(self, x):
        with tf.variable_scope('transformer_encoder', reuse=tf.AUTO_REUSE):
            enc_input = self.x
            enc_input *= self.dim_model**0.5
            enc_input += self.positional_encoding(enc_input, self.max_sen_len)
            enc_input = tf.layers.dropout(enc_input, self.dropout_rate)
            for i in range(self.num_enc_blocks):
                with tf.variable_scope('encoder ' + str(i), reuse=tf.AUTO_REUSE):
                    enc_input = self.multihead_attention(enc_input)
                    enc_input = self.feed_forward(enc_input, [self.dim_feed_for, self.dim_model])
            return enc_input
    
    def multihead_attention(self, enc_input):
        dim_model = enc_input.get_shape().as_list()[-1]
        with tf.variable_scope('multi_head_attention', reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(enc_input, d_model, use_bias=False)
            K = tf.layers.dense(enc_input, d_model, use_bias=False)
            V = tf.layers.dense(enc_input, d_model, use_bias=False)
            Q = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = self.scaled_dot_product_attention(Q, K, V)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += enc_input
            outputs = self.layer_normalization(outputs)
            return outputs
    
    def scaled_dot_product_attention(self, Q, K, V):
        with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            outputs /= d_k**0.5
            outputs = self.mask(outputs, Q, K, is_key=True)
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            outputs = self.mask(outputs, Q, K)
            outputs = tf.layers.dense(outputs, rate=self.dropout_rate)
            outputs = tf.matmul(outputs, V)
            return outputs
    
    def mask(self, inputs, Q, K, is_key=False):
        padding_num = -2**32 + 1
        if is_key:
            masks = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))
            masks = tf.expand_dims(masks, 1)
            masks = tf.tile(masks, [1, tf.shape(Q)[1], 1])
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        else:
            masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
            masks = tf.expand_dims(masks, -1)
            masks = tf.tile(masks, [1, 1, tf.shape(K)[1]])
            outputs = inputs * masks
        return outputs
        
    def feed_forward(self, inputs, num_units):
        with tf.variable_scope('feed_forward', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1])
            outputs += inputs
            outputs = self.layer_normalization(outputs)
            return outputs
    
    def layer_normalization(self, inputs, epsilon=1e-8):
        with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.zeros_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
            return outputs
            
            
    def positional_encoding(self, enc_input, max_len):
        E = enc_input.get_shape().as_list()[-1]
        N, T = tf.shape(enc_input)[0], tf.shape(enc_input)[1]
        with tf.variable_scope('positional_encoding', reuse=tf.AUTO_REUSE):
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
            position_enc = np.array([
            [pos / np.power(1000, (i - i%2)/E) for i in range(E)]
            for pos in range(max_len)
            ])
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            #Masking by default
            outputs = tf.where(tf.equal(enc_input, 0), enc_input, outputs)
            return tf.to_float(outputs)
            
        