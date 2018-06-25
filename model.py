from __future__ import print_function
from __future__ import division


import tensorflow as tf

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
            One can use attributes to read/write dictionary content.
        '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self



def conv2d(input_, output_dim, k_h, k_w, name = 'conv2d'):
    '''
         :param input_: shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]
         :param output_dim: [kernel_features], which is # of kernels with this width
         :param k_h: 1
         :param k_w: kernel width, n-grams
         :param name: name scope
         :return: shape = [reduced_length, output_dim]
    '''

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def linear(input_, output_size, scope = None):
    '''
        input needs to be a 2D matrix, with its shape[1] is valid
        :param intput_: shape = [batch_size * num_unroll_steps, sum(kernel_features)]
        :param output_size: shape = [batch_size * num_unroll_steps, sum(kernel_features)] cause it needs to plus the original input
        :param scope: variable scope
        :return:

    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    :param input_: a tensor or a list of 2D, batch x n, Tensors
    :param output_size: second dim of W[i], output_size
    :param scope: variable scope for the created scope, default linaer!
    :return: A 2D Tensor with shape [batch, output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.

    '''



    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))

    input_size = shape[1]

    with tf.variable_scope(scope or 'SimpleLinear'):

        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype= input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype = input_.dtype)

    return tf.matmul(input_, matrix, transpose_b= True)  + bias_term


def highway(input_, size, num_layers = 1, bias = -2.0, f = tf.nn.relu, scope = 'Highway'):
    '''

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

        :param input_: [batch_size * num_unroll_steps, sum(kernel_features)]
        :param size: this is the output_dim of the kernels, which should be matched for the input_ dim shape[1]
        :param num_layers: how many highway layers you want
        :param bias: transform gate bias
        :param f: linear activation
        :param scope: highway
        :return:
            t = sigmoid(W_Ty + b_T)
            z = t * f(W_Hy + b_H) + (1. - t) * y
    '''


    with tf.variable_scope(scope):

        for idx in range(num_layers):

            g = f(linear(input_, size, scope = 'highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope = 'highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_

            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, scope = 'TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)

        :param input_: shape = [batch_size * num_unroll_steps, max_sent_length, embed_size]
        :param kernels: n-grams
        :param kernel_features: how many features we want to use for each n-gram kernel
        :param scope: variable scope
        :return:
    '''

    assert len(kernels) == len(kernel_features)

    max_sent_length = input_.get_shape()[1]


    input_ = tf.expand_dims(input_, 1)
    # input_, shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]

    layers = []

    with tf.variable_scope(scope):

        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):

            reduced_length = max_sent_length - kernel_size + 1

            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name= 'kernel_%d' % kernel_size)
            # conv, shape = [batch_size * num_unroll_steps, 1, reduced_length, kernel_feature_size]


            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1],  [1, 1, 1, 1], 'VALID')
            # pool, shape = [batch_size * num_unroll_steps, 1, 1, kernel_feature_size]

            layers.append(tf.squeeze(pool, [1, 2]))

    if len(kernels) > 1:
        output = tf.concat(layers, 1) # shape = [batch_size * num_unroll_steps, sum(kernel_features)]
    else:
        output = layers[0]

    return output

def inference_graph(word_vocab_size = 1738,
                    training = True,
                    kernels = [1, 2, 3, 4, 5, 6, 7],
                    kernel_features = [25, 50, 75, 100, 125, 150, 175],
                    rnn_size = 128,
                    dropout = 0.0,
                    num_rnn_layers = 1,
                    num_highway_layers = 1,
                    num_unroll_steps = 60,
                    max_sent_length = 30,
                    #batch_size = 20,
                    embed_size = 120):

        '''

        :param training:
        :param kernels:
        :param kernel_features:
        :param rnn_size:
        :param dropout:
        :param num_rnn_layers:
        :param num_highway_layers:
        :param num_unroll_steps:
        :param max_sent_length:
        :param batch_size:
        :param embed_size:
        :return:
        '''



        assert len(kernels) == len(kernel_features)

        input_ = tf.placeholder(tf.int32, shape = [None, num_unroll_steps, max_sent_length], name = 'input')

        batch_size = tf.placeholder(tf.int32, shape = [], name = 'batch_size')


        #print(input_.get_shape())

        sequence_length = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')

        with tf.variable_scope('Embedding'):

            word_embedding = tf.get_variable('word_embedding', shape = [word_vocab_size, embed_size])

            clear_word_embedding_padding = tf.scatter_update(word_embedding, [0],
                                                             tf.constant(0.0, shape = [1, embed_size]))


            input_embedded = tf.nn.embedding_lookup(word_embedding, input_)

            #print(input_embedded.get_shape())

            input_embedded = tf.reshape(input_embedded, [-1, max_sent_length, embed_size])
            #print(input_embedded.get_shape())
            '''Characters embedded'''

            '''Second, apply convolution'''

        input_cnn = tdnn(input_embedded, kernels, kernel_features)

        #print(input_cnn)

        if num_highway_layers > 0:
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers= num_highway_layers)


        with tf.variable_scope('LSTM'):

            def create_cnn_cell():

                cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple = True, forget_bias = 0.0, reuse = False)

                if dropout > 0.0:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1. - dropout)

                return cell

            if num_rnn_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([create_cnn_cell() for _ in range(num_rnn_layers)])

            else:

                cell = create_cnn_cell()


            #print(input_cnn)

            initial_rnn_state = cell.zero_state(batch_size, dtype= 'float32')

            input_cnn = tf.reshape(input_cnn, [-1, num_unroll_steps, input_cnn.get_shape()[-1]])


            outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, input_cnn, initial_state= initial_rnn_state, sequence_length = sequence_length,
                                           dtype= 'float32')

            #outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.reshape(outputs, shape = [-1, outputs.get_shape()[-1]])
            predictions = tf.tanh(linear(outputs, 3, scope = 'prediction_linear'))
            #print(predictions)

        return adict(input = input_,
                 clear_word_embedding_padding = clear_word_embedding_padding,
                 input_embedded = input_embedded, # this is the embedded representation of input batches
                 input_cnn = input_cnn, # this is the thing of word representations before fed into rnn
                 initial_rnn_state = initial_rnn_state, # rnn initial state
                 final_rnn_state = final_rnn_state, # rnn final state
                 rnn_outputs = outputs, # rnn hidden states
                 predictions = predictions,
                 sequence_length = sequence_length,
                 batch_size = batch_size)



def loss_graph(predictions, groundtruth):

    with tf.variable_scope('Loss'):

        groundtruth = tf.reshape(groundtruth, shape = [-1, 3])

        loss = {0 : 0.0, 1: 0.0, 2: 0.0}

        for i in [0, 1, 2]:

            pred_mean, pred_var = tf.nn.moments(predictions[:, i], [0])
            gt_mean, gt_var = tf.nn.moments(groundtruth[:, i], [0])

            mean_cent_prod = tf.reduce_mean((predictions[:,i] - pred_mean) * (groundtruth[:,i] - gt_mean))

            loss[i] = 1. - 2. * mean_cent_prod / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

    return adict(
        loss_arousal = loss[0],
        loss_valence = loss[1],
        loss_liking = loss[2]
    )

def eval_metric_graph():
    with tf.variable_scope('Eval_LOSS'):
        pred = tf.placeholder(tf.float32, [None, 3], name = 'pred')
        label = tf.placeholder(tf.float32, [None, 3], name= 'label')

        metric = {0: 0.0, 1: 0.0, 2: 0.0}

        for i in [0, 1, 2]:

            pred_mean, pred_var = tf.nn.moments(pred[:, i], [0])
            gt_mean, gt_var = tf.nn.moments(label[:, i], [0])

            mean_cent_prod = tf.reduce_mean((pred[:,i] - pred_mean) * (label[:,i] - gt_mean))
            #cov = tf.contrib.metrics.streaming_covariance(pred[:, i], label[:, i])[0]



            metric[i] = 2. * mean_cent_prod / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

    return adict(
        eval_predictions = pred,
        eval_labels = label,
        eval_metric_arousal = metric[0],
        eval_metric_valence = metric[1],
        eval_metric_liking = metric[2]
    )