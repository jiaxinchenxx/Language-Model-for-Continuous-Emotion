import tensorflow as tf

import os
import time
import numpy as np
import tensorflow as tf

import model
import data_utils as dl


PATH = 'D:\PythonProject\AVEC'
LOGGING_PATH = PATH + '\check_points\log.txt'
SAVE_PATH = PATH + '\check_points\word_CNN_model.ckpt'

flags = tf.flags

# path related
flags.DEFINE_string('load_model',           SAVE_PATH,           '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')


# model params
flags.DEFINE_integer('word_vocab_size',     1738,                               'size of vocabulary')
flags.DEFINE_integer('rnn_size',            256,                                'size of LSTM cell')
flags.DEFINE_integer('highway_layers',      1,                                  'size of highway layers')
flags.DEFINE_integer('word_embed_size',     256,                                'embed_size')
flags.DEFINE_string ('kernels',              '[1,2,3,4,5,6,7]',                  'CNN kernel width')
flags.DEFINE_string ('kernel_features',     '[25,50,75,100,125,150,175]',       'CNN kernel num')
flags.DEFINE_integer('rnn_layers',          1,                                  'num of layers of RNN')
flags.DEFINE_float  ('dropout',             0.0,                                'dropout')

# optimization

flags.DEFINE_integer('num_unroll_steps',    60,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          464,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_sent_length',     30,   'maximum word length')

FLAGS = flags.FLAGS


def evaluation():

    assert FLAGS.load_model != None

    input_tensors, label_tensors, seq_tensors = dl.make_batches()

    test_reader = dl.DataReader(input_tensors['Test'], label_tensors['Test'], seq_tensors['Test'],
                                FLAGS.batch_size, FLAGS.num_unroll_steps)

    labels = tf.placeholder(tf.float32, [None, FLAGS.num_unroll_steps, 3], name='labels')


    test_model = model.inference_graph(word_vocab_size= FLAGS.word_vocab_size,
                                        kernels= eval(FLAGS.kernels),
                                        kernel_features= eval(FLAGS.kernel_features),
                                        rnn_size= FLAGS.rnn_size,
                                        dropout= FLAGS.dropout,
                                        num_rnn_layers= FLAGS.rnn_layers,
                                        num_highway_layers= FLAGS.highway_layers,
                                        num_unroll_steps= FLAGS.num_unroll_steps,
                                        max_sent_length= FLAGS.max_sent_length,
                                        batch_size= FLAGS.batch_size,
                                        embed_size= FLAGS.word_embed_size)

    predictions = test_model.predictions

    print(predictions)

    losses = model.loss_graph(predictions, labels)

    loss_arousal = losses.loss_arousal
    loss_valence = losses.loss_valence
    loss_liking = losses.loss_liking

    metric_arousal = 1. - loss_arousal
    metric_valence = 1. - loss_valence
    metric_liking = 1. - loss_liking

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print('load model %s ...' % SAVE_PATH)
        saver.restore(sess, SAVE_PATH)
        print('done!')

        metric = []

        for minibatch in test_reader.iter():

            x, y = minibatch

            m_arousal, m_valence, m_liking = sess.run([metric_arousal, metric_valence, metric_liking],
                                    feed_dict={
                                        test_model.input:x,
                                        labels:y
                                    })

            metric.append([m_arousal, m_valence, m_liking])

        metric = np.mean(np.array(metric), axis= 0)

        print('Test Reuslt: arousal: %.4f -- valence: %.4f -- liking: %.4f'
              % (metric[0], metric[1], metric[2]))


if __name__ == '__main__':
    evaluation()
