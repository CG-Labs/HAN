import numpy as np
import tensorflow as tf

conv1d = tf.keras.layers.Conv1D


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # Ensure seq is 3-dimensional
        seq_shape = tf.shape(seq)
        # Assert that the input tensor seq has a rank of 3
        tf.debugging.assert_equal(tf.rank(seq), 3, message='Input tensor seq must be 3-dimensional')
        # Assuming the input tensor seq has a shape of (batch_size, num_nodes, num_features)
        # Check if the seq tensor is already 3-dimensional and has the correct shape
        if tf.rank(seq) == 3 and seq_shape[1] == bias_mat.shape[0]:
            # If the shape is already correct, proceed without reshaping
            seq_fts = conv1d(out_sz, 1, use_bias=False)(seq)
        else:
            # If the shape is not correct, reshape seq to have three dimensions: (batch_size, num_nodes, feature_size)
            # Calculate the feature size assuming the last dimension of seq is the total number of features
            total_features = seq_shape[-1]
            feature_size = total_features // bias_mat.shape[0]
            seq = tf.reshape(seq, (-1, bias_mat.shape[0], feature_size))
            seq_fts = conv1d(out_sz, 1, use_bias=False)(seq)
        # Rest of the function remains unchanged
        f_1 = conv1d(1, 1)(seq_fts)
        f_2 = conv1d(1, 1)(seq_fts)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        logits = tf.debugging.check_numerics(logits, "logits contains NaN or Inf")
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        coefs = tf.debugging.check_numerics(coefs, "coefs after softmax contains NaN or Inf")
        coefs_non_zero = tf.math.count_nonzero(coefs)
        tf.debugging.assert_positive(coefs_non_zero, message="Attention coefficients (coefs) are all zeros after softmax")

        # Reshape seq_fts to be a 2D tensor for matrix multiplication
        seq_fts = tf.squeeze(seq_fts, axis=1)
        # Reshape coefs to be a 2D tensor for matrix multiplication
        coefs = tf.squeeze(coefs, axis=0)

        vals = tf.matmul(coefs, seq_fts)
        vals = tf.debugging.check_numerics(vals, "vals after matmul contains NaN or Inf")
        vals_non_zero = tf.math.count_nonzero(vals)
        tf.debugging.assert_positive(vals_non_zero, message="Result of matrix multiplication (vals) is all zeros")

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        ret = tf.keras.layers.Add()([vals, tf.zeros_like(vals)])
        ret = tf.debugging.check_numerics(ret, "ret contains NaN or Inf after bias_add")

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)


def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    adj_mat = 1.0 - bias_mat / -1e9
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq = tf.expand_dims(seq, axis=1)  # Add a sequence length dimension
        seq_fts = conv1d(out_sz, 1, use_bias=False)(seq)

        logits = adj_mat
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.keras.layers.Add()([vals, tf.zeros_like(vals)])

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        return activation(ret)


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = conv1d(out_sz, 1, use_bias=False)(seq)

        f_1 = conv1d(1, 1)(seq_fts)
        f_2 = conv1d(1, 1)(seq_fts)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2, [0, 2, 1]))
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.keras.layers.Add()([vals, tf.zeros_like(vals)])

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        return activation(ret)


def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2]

    w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
