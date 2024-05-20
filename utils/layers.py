import numpy as np
import tensorflow as tf

conv1d = tf.keras.layers.Conv1D

class BiasAddLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BiasAddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
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
        return tuple(dim for dim in input_shape if dim is not None and dim != 1)

class BroadcastToLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BroadcastToLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BroadcastToLayer, self).build(input_shape)

    def call(self, inputs):
        seq_fts, bias_mat_reshaped = inputs
        # Ensure bias_mat_reshaped is broadcasted to match the shape of seq_fts
        seq_fts_shape = tf.shape(seq_fts)
        bias_mat_broadcasted = tf.broadcast_to(bias_mat_reshaped, [seq_fts_shape[0], seq_fts_shape[1], seq_fts_shape[2]])
        return bias_mat_broadcasted

    def compute_output_shape(self, input_shape):
        # The output shape should match the shape of seq_fts
        return input_shape[0]

class ReshapeBiasMatLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReshapeBiasMatLayer, self).__init__(**kwargs)

    def call(self, inputs):
        seq_fts, bias_mat = inputs
        # Get the dynamic shape of seq_fts
        seq_fts_shape = tf.shape(seq_fts)
        # Calculate the total number of elements in bias_mat
        total_elements_bias_mat = tf.size(bias_mat)
        # Calculate the side length of the square matrix after reshaping
        # The side length should be the square root of the total number of elements in bias_mat
        # divided by the batch size of seq_fts, ensuring the total number of elements remains the same
        side_length = tf.cast(tf.sqrt(tf.cast(total_elements_bias_mat / seq_fts_shape[0], tf.float32)), tf.int32)
        # Use tf.cond to perform the conditional check and assignment
        side_length_corrected = tf.cond(tf.math.equal(side_length * side_length * seq_fts_shape[0], total_elements_bias_mat),
                                        lambda: side_length,
                                        lambda: side_length + 1)
        # Reshape bias_mat to have the shape [batch_size, side_length_corrected, side_length_corrected]
        bias_mat_reshaped = tf.reshape(bias_mat, [seq_fts_shape[0], side_length_corrected, side_length_corrected])
        return bias_mat_reshaped

    def compute_output_shape(self, input_shape):
        # The output shape should match the shape of seq_fts
        return input_shape[0]

class PrintShapeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Use tf.print to output the shapes of the tensors
        tf.print("seq_fts_shape:", tf.shape(inputs[0]))
        tf.print("bias_mat_shape:", tf.shape(inputs[1]))
        tf.print("bias_mat_reshaped_shape:", tf.shape(inputs[2]))

        # Return the original input tensor
        return inputs

# Custom Keras layer to extract and return the shapes of tensors
class ExtractShapesLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Extract the shapes of each tensor in the inputs list
        shapes = [tf.shape(input_tensor) for input_tensor in inputs]
        return shapes

# Custom layer to perform addition of two tensors
class AddLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Perform element-wise addition of two tensors with compatible shapes
        # Ensure that bias_mat_reshaped is broadcasted to match the shape of seq_fts
        seq_fts, bias_mat_reshaped = inputs
        seq_fts_shape = tf.shape(seq_fts)
        bias_mat_broadcasted = tf.broadcast_to(bias_mat_reshaped, seq_fts_shape)
        return tf.add(seq_fts, bias_mat_broadcasted)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as the shape of either input tensor since they are compatible
        return input_shape[0]

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False, return_shapes=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.keras.layers.Dropout(1.0 - in_drop)(seq, training=True)
        if len(seq.shape) == 4 and seq.shape[-1] == 1:
            seq = tf.reshape(seq, (-1, seq.shape[1], seq.shape[2]))
        elif len(seq.shape) == 2:
            seq = tf.reshape(seq, (-1, seq.shape[0], seq.shape[1]))
        seq_fts = conv1d(filters=seq.shape[-1], kernel_size=1, use_bias=False)(seq)

        f_1 = conv1d(filters=1, kernel_size=1)(seq_fts)
        f_2 = conv1d(filters=1, kernel_size=1)(seq_fts)

        logits = f_1 + tf.keras.layers.Permute((2, 1))(f_2)
        leaky_relu = tf.keras.layers.LeakyReLU()(logits)

        # Reshape bias_mat to match the shape required for broadcasting with seq_fts
        reshape_bias_mat_layer = ReshapeBiasMatLayer()
        bias_mat_reshaped = reshape_bias_mat_layer([seq_fts, bias_mat])

        # Use PrintShapeLayer to print the shapes
        print_shape_layer = PrintShapeLayer()
        # Print shapes after reshaping bias_mat
        shapes_after_reshape = print_shape_layer([seq_fts, bias_mat, bias_mat_reshaped])
        tf.print("Shapes after reshape:", shapes_after_reshape)

        try:
            add_layer = AddLayer()
            coefs = tf.keras.layers.Softmax(axis=-1)(add_layer([leaky_relu, bias_mat_reshaped]))
            # Print shapes after broadcasting
            shapes_after_broadcast = print_shape_layer([seq_fts, bias_mat_reshaped, coefs])
            tf.print("Shapes after broadcast:", shapes_after_broadcast)
        except Exception as e:
            tf.print("Exception during broadcasting: ", e)
            tf.print("leaky_relu shape at exception:", tf.shape(leaky_relu))
            tf.print("bias_mat_reshaped shape at exception:", tf.shape(bias_mat_reshaped))
            raise e

        if coef_drop != 0.0:
            coefs = tf.keras.layers.Dropout(1.0 - coef_drop)(coefs, training=True)
        if in_drop != 0.0:
            seq_fts = tf.keras.layers.Dropout(1.0 - in_drop)(seq_fts, training=True)

        vals = tf.keras.layers.Dot(axes=(1, 1))([coefs, seq_fts])

        bias_add_layer = BiasAddLayer()
        ret = bias_add_layer(vals)

        # Apply residual connection if specified
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)
            else:
                ret = ret + seq

        # Instantiate ELULayer to apply ELU activation function
        elu_layer = ELULayer()
        ret = elu_layer(ret)

        # If return_shapes is True, use ExtractShapesLayer to capture and return shapes
        if return_shapes:
            # Instantiate ExtractShapesLayer and use it to extract shapes
            extract_shapes_layer = ExtractShapesLayer()
            shapes = extract_shapes_layer([seq_fts, bias_mat, bias_mat_reshaped])
            return ret, shapes
        else:
            if return_coef:
                return ret, coefs
            else:
                return ret

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

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)
            else:
                seq_fts = ret + seq

        elu_layer = ELULayer()
        return elu_layer(ret)

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.keras.layers.Dropout(1.0 - in_drop)(seq, training=True)

        seq_fts = conv1d(filters=out_sz, kernel_size=1, use_bias=False)(seq)

        f_1 = conv1d(filters=1, kernel_size=1)(seq_fts)
        f_2 = conv1d(filters=1, kernel_size=1)(seq_fts)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2, [0, 2, 1]))
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

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.keras.backend.bias_add(vals)

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(filters=ret.shape[-1], kernel_size=1)(seq)
            else:
                seq_fts = ret + seq

        return activation(ret)

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
