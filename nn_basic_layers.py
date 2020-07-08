import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers

"""
Predefine all necessary layer for the R-CNN
"""
def conv_bn_relu(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, is_training=True, bias=False, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        if bias == True:
            biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)

        if bias == True:
            # Add biases
            feat = tf.nn.bias_add(conv, biases)
            feat = tf.reshape(feat, tf.shape(conv))
        else:
            feat = conv

        # batch norm
        feat_bn = tf.contrib.layers.batch_norm(feat, center=True, scale=True, is_training=is_training)

        return tf.nn.relu(feat_bn)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        bias = tf.nn.bias_add(conv, biases)
        bias = tf.reshape(bias, tf.shape(conv))
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)  # Apply ReLu non linearity
            return relu
        else:
            return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def avg_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

# new version of tensorflow has the new way to create multiplayer RNN
def bidirectional_recurrent_layer(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    if (nlayer == 1):
        fw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
        bw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
    else:
        fw_cell_ = []
        bw_cell_ = []
        for i in range(nlayer):
            fw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
            bw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
        fw_cell = tf.contrib.rnn.MultiRNNCell(cells=fw_cell_, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell(cells=bw_cell_, state_is_tuple=True)
    # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

# new version of tensorflow has the new way to create multiplayer RNN
def bidirectional_recurrent_layer_output(fw_cell, bw_cell, input_layer, sequence_len, scope=None):
    ((fw_outputs,
      bw_outputs),
     (fw_state,
      bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                    cell_bw=bw_cell,
                                                    inputs=input_layer,
                                                    sequence_length=sequence_len,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    scope=scope))
    outputs = tf.concat((fw_outputs, bw_outputs), 2)

    def concatenate_state(fw_state, bw_state):
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
            return state
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat((fw_state, bw_state), 1,
                              name='bidirectional_concat')
            return state
        elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                      len(fw_state) == len(bw_state)):
            # multilayer
            state = tuple(concatenate_state(fw, bw)
                          for fw, bw in zip(fw_state, bw_state))
            return state

        else:
            raise ValueError(
                'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state


def self_attention(inputs, attention_size, scaled_=True, masked_=False, name="self-attention"):
    inputs_shape = inputs.shape  # (B,seq_len, ndim)
    seq_len = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    ndim = inputs_shape[2].value  # hidden size of the RNN layer
    with tf.variable_scope(name) as scope:
        Q = tf.layers.dense(inputs, attention_size)  # [batch_size, sequence_length, hidden_dim]
        K = tf.layers.dense(inputs, attention_size)  # [batch_size, sequence_length, hidden_dim]
        V = tf.layers.dense(inputs, ndim)  # [batch_size, sequence_length, ndim]

        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]
        if masked_:
            raise NotImplementedError

        attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
        output = tf.matmul(attention, V)  # [batch_size, sequence_length, n_classes]

    return output



