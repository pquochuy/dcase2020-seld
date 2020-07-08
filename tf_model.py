import tensorflow as tf
from nn_basic_layers import *

class SELDnet(object):
    def __init__(self, params, in_shape, out_shape_sed, out_shape_doa):
        self.params = params
        self.in_shape = in_shape
        self.out_shape_sed = out_shape_sed
        self.out_shape_doa = out_shape_doa
        self.input_x_mel = tf.placeholder(tf.float32, [None, self.in_shape[0], self.in_shape[1], self.in_shape[2]], name="input_x_mel")
        self.input_y_sed = tf.placeholder(tf.float32, [None, self.out_shape_sed[0], self.out_shape_sed[1]], name="input_y_sed")
        self.input_y_doa = tf.placeholder(tf.float32, [None, self.out_shape_doa[0], self.out_shape_doa[1]], name="input_y_doa")
        self.dropout_keep_prob_cnn = tf.placeholder(tf.float32, name="dropout_keep_prob_cnn")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.dropout_keep_prob_dnn = tf.placeholder(tf.float32, name="dropout_keep_prob_dnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining')  # idicate training for batch normmalization

        self.frame_seq_len = tf.placeholder(tf.int32, [None])  # for the dynamic RNN

        self.rnn_out = self.construct_crnn(self.input_x_mel, name='mel_branch')
        self.rnn_out = tf.reshape(self.rnn_out, [-1, 2*self.params['rnn_hidden_size']])

        with tf.variable_scope("sed_output"):
            fc1 = fc(self.rnn_out, 2*self.params['rnn_hidden_size'], self.params['dnn_size'], name="fc1", relu=True)
            fc1 = dropout(fc1, self.dropout_keep_prob_dnn)
            fc2 = fc(fc1, self.params['dnn_size'], self.params['dnn_size'], name="fc2", relu=True)
            fc2 = dropout(fc2, self.dropout_keep_prob_dnn)
            self.score_sed = fc(fc2, self.params['dnn_size'], self.out_shape_sed[-1], name="output", relu=False)
        self.sed_pred_2d = tf.sigmoid(self.score_sed)
        self.sed_pred = tf.reshape(self.sed_pred_2d, [-1, self.out_shape_sed[0], self.out_shape_sed[1]])
        self.sed_loss = self.sed_loss_regression(self.sed_pred_2d, tf.reshape(self.input_y_sed,[-1, self.out_shape_sed[-1]]))

        with tf.variable_scope("doa_output"):
            fc1 = fc(self.rnn_out, 2 * self.params['rnn_hidden_size'], self.params['dnn_size'], name="fc1", relu=True)
            fc1 = dropout(fc1, self.dropout_keep_prob_dnn)
            fc2 = fc(fc1, self.params['dnn_size'], self.params['dnn_size'], name="fc2", relu=True)
            fc2 = dropout(fc2, self.dropout_keep_prob_dnn)
            self.score_doa = fc(fc2, self.params['dnn_size'], self.out_shape_doa[-1], name="output", relu=False)
        self.doa_pred_2d = tf.tanh(self.score_doa)
        self.doa_pred = tf.reshape(self.doa_pred_2d, [-1, self.out_shape_doa[0], self.out_shape_doa[1]])
        # this is soft mask
        mask = tf.reshape(self.input_y_sed, [-1, self.out_shape_sed[-1]])
        mask = tf.concat((mask, mask, mask), axis=-1)
        self.doa_loss = self.doa_loss_regression(self.doa_pred_2d,
                                                 tf.reshape(self.input_y_doa, [-1, self.out_shape_doa[-1]]),
                                                 mask)

        self.output_loss = self.params['loss_weights'][0]*self.sed_loss \
                           + self.params['loss_weights'][1]*self.doa_loss

        # add on regularization except the filter bank layer
        with tf.name_scope("l2_loss"):
            vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])
        self.loss = self.output_loss + self.params['l2_reg_lambda'] * l2_loss

    def construct_crnn(self, input, name):
        with tf.variable_scope(name):
            # (-1, 300, 64, 7)
            conv1 = conv_bn_relu(input, 3, 3, 64, 1, 1, is_training=self.istraining, padding='SAME', name='conv1')
            print(conv1.get_shape())
            # (-1, 300, 64, 64)
            conv2 = conv_bn_relu(conv1, 3, 3, 64, 1, 1, is_training=self.istraining, padding='SAME', name='conv2')
            print(conv2.get_shape())
            # (-1, 300, 64, 64)
            pool2 = max_pool(conv2, 5, 2, 5, 2, padding='VALID', name='pool1')
            print(pool2.get_shape())
            # (-1, 60, 32, 64)
            dropout2 = dropout(pool2, self.dropout_keep_prob_cnn)


            # (-1, 60, 32, 64)
            conv3 = conv_bn_relu(dropout2, 3, 3, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv3')
            print(conv3.get_shape())
            # (-1, 60, 32, 128)
            pool3 = max_pool(conv3, 1, 2, 1, 2, padding='VALID', name='pool3')
            dropout3 = dropout(pool3, self.dropout_keep_prob_cnn)
            # (-1, 60, 16, 128)

            # (-1, 60, 16, 128)
            conv4 = conv_bn_relu(dropout3, 3, 3, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv4')
            print(conv4.get_shape())
            # (-1, 60, 16, 256)
            pool4 = max_pool(conv4, 1, 2, 1, 2, padding='VALID', name='pool4')
            dropout4 = dropout(pool4, self.dropout_keep_prob_cnn)
            # (-1, 60, 8, 128)

            # (-1, 60, 8, 128)
            conv5 = conv_bn_relu(dropout4, 3, 3, 256, 1, 1, is_training=self.istraining, padding='SAME', name='conv5')
            print(conv5.get_shape())
            # (-1, 60, 8, 256)
            pool5 = max_pool(conv5, 1, 2, 1, 2, padding='VALID', name='pool5')
            dropout5 = dropout(pool5, self.dropout_keep_prob_cnn)
            # (-1, 60, 4, 128)

            # (-1, 60, 4, 128)
            conv6 = conv_bn_relu(dropout5, 3, 3, 256, 1, 1, is_training=self.istraining, padding='SAME', name='conv6')
            print(conv6.get_shape())
            # (-1, 60, 4, 256)
            pool6 = max_pool(conv6, 1, 2, 1, 2, padding='VALID', name='pool6')
            dropout6 = dropout(pool6, self.dropout_keep_prob_cnn)
            # (-1, 60, 2, 256)

            conv_out = tf.reshape(dropout6, [-1, self.out_shape_sed[0], 2 * 256])

            # bidirectional frame-level recurrent layer
            with tf.variable_scope("frame_rnn_layer") as scope:
                fw_cell, bw_cell = bidirectional_recurrent_layer(self.params['rnn_hidden_size'],
                                                                 self.params['rnn_nb_layer'],
                                                                 input_keep_prob=1.0, # we have dropouted in the convolutional layer
                                                                 output_keep_prob=self.dropout_keep_prob_rnn)
                rnn_out, rnn_state = bidirectional_recurrent_layer_output(fw_cell, bw_cell,
                                                                          conv_out, self.frame_seq_len,
                                                                          scope=scope)
                print('rnn_out')
                print(rnn_out.get_shape())
                # output shape (-1, seq_len, nhidden*2)

            # self-attention
            rnn_out = self_attention(rnn_out, self.params['attention_size'], name="self-attention")
            print('rnn_attention_out')
            print(rnn_out.get_shape())
            # output shape (-1, seq_len, nhidden*2)
        return rnn_out

    def sed_loss_regression(self, pred, gt):
        # instead of cross-entropy loss, we use mse loss (regression) here
        sed_loss = tf.reduce_mean(tf.square(pred - gt)) # mean in all dimensions
        return sed_loss

    def sed_loss_classification(self, pred, gt):
        # instead of cross-entropy loss, we use mse loss (regression) here
        sed_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)
        sed_loss = tf.reduce_sum(sed_loss, axis=[1])
        sed_loss = tf.reduce_mean(sed_loss) # mean in all dimensions
        return sed_loss

    def doa_loss_regression(self, pred, gt, mask):
        doa_loss = tf.square(pred - gt)
        doa_loss = tf.multiply(doa_loss, mask) # mask here
        doa_loss = tf.reduce_sum(doa_loss)/tf.reduce_sum(mask) # mean in all dimensions
        return doa_loss

