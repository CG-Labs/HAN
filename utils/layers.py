import numpy as np
import tensorflow as tf

conv1d = tf.keras.layers.Conv1D

class BiasAddLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BiasAddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Set the bias size to match the last dimension of the input shape
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    name='bias')
        super(BiasAddLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.keras.backend.bias_add(inputs, self.bias)

class ELULayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.elu(inputs)

class SqueezeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.squeeze(inputs)

    def compute_output_shape(self, input_shape):
        # Remove all singleton dimensions from the input shape, skip None dimensions
        return tuple(dim for dim in input_shape if dim is not None and dim != 1)

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
               return_coef=False):
    """[summary]

    [description]

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.keras.layers.Dropout(1.0 - in_drop)(seq, training=True)
        # Ensure seq is three-dimensional before passing to conv1d
        # Reshape seq to have 3 dimensions (batch_size, steps, input_dim)
        if len(seq.shape) == 4 and seq.shape[-1] == 1:
            seq = tf.reshape(seq, (-1, seq.shape[1], seq.shape[2]))
        elif len(seq.shape) == 2:
            seq = tf.reshape(seq, (-1, seq.shape[0], seq.shape[1]))
        seq_fts = conv1d(filters=seq.shape[-1], kernel_size=1, use_bias=False)(seq)

        f_1 = conv1d(filters=1, kernel_size=1)(seq_fts)
        f_2 = conv1d(filters=1, kernel_size=1)(seq_fts)

        logits = f_1 + tf.keras.layers.Permute((2, 1))(f_2)
        leaky_relu = tf.keras.layers.LeakyReLU()(logits)
        coefs = tf.keras.layers.Softmax(axis=-1)(leaky_relu + tf.broadcast_to(bias_mat, tf.shape(leaky_relu)))

        if coef_drop != 0.0:
            coefs = tf.keras.layers.Dropout(1.0 - coef_drop)(coefs, training=True)
        if in_drop != 0.0:
            seq_fts = tf.keras.layers.Dropout(1.0 - in_drop)(seq_fts, training=True)

        vals = tf.keras.layers.Dot(axes=(1, 1))([coefs, seq_fts])

        # Replace the direct bias_add call with an instance of the custom layer
        bias_add_layer = BiasAddLayer()
        ret = bias_add_layer(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)  # activation
            else:
                seq_fts = ret + seq

        # Replace the direct call to the activation function with an instance of ELULayer
        elu_layer = ELULayer()
        if return_coef:
            return elu_layer(ret), coefs
        else:
            return elu_layer(ret)  # Apply ELU activation using the custom layer

def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    adj_mat = 1.0 - bias_mat / -1e9
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.keras.layers.Dropout(1.0 - in_drop)(seq, training=True)
        seq_fts = conv1d(filters=out_sz, kernel_size=1, use_bias=False)(seq)


        logits = adj_mat
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.keras.layers.Dropout(1.0 - coef_drop)(coefs, training=True)
        if in_drop != 0.0:
            seq_fts = tf.keras.layers.Dropout(1.0 - in_drop)(seq_fts, training=True)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.keras.backend.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)  # activation
            else:
                seq_fts = ret + seq

        # Replace the direct call to the activation function with an instance of ELULayer
        elu_layer = ELULayer()
        return elu_layer(ret)  # Apply ELU activation using the custom layer

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.keras.layers.Dropout(1.0 - in_drop)(seq, training=True)

        seq_fts = conv1d(filters=out_sz, kernel_size=1, use_bias=False)(seq)

        # simplest self-attention possible
        f_1 = conv1d(filters=1, kernel_size=1)(seq_fts)
        f_2 = conv1d(filters=1, kernel_size=1)(seq_fts)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat *
                               tf.transpose(f_2, [0, 2, 1]))
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.keras.layers.Dropout(1.0 - coef_drop)(coefs.values, training=True),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.keras.layers.Dropout(1.0 - in_drop)(seq_fts, training=True)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.keras.backend.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
#                                                      time_major=False,
#                                                      return_alphas=True)
def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
