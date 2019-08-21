import tensorflow as tf

class Transformer:
    
    def __init__(self, embedding_size, max_sen_len, dim_k, dim_q, dim_model):
        self.x = tf.placeholder(shape=[None, max_sent_len, embedding_size])
        
    def encode(self, x):
        with tf.variable_scope('transformer_encoder', reuse=tf.AUTO_REUSE):
            enc_input = self.x
            enc_input *= self.dim_model**0.5
            enc_input += self.positional_encoding(enc_input, self.max_sen_len)
            enc_input = tf.layers.dropout(enc_input, self.dropout_rate)
            
    def positional_encoding(enc_input, max_len):
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
            
        