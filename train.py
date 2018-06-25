import tensorflow as tf

import os
import time
import numpy as np
import tensorflow as tf

import model
import data_utils_advanced as dl

PATH = 'D:\PythonProject\AVEC'
LOGGING_PATH = PATH + '\check_points\log.txt'
SAVE_PATH = PATH + '\check_points\word_CNN_model.ckpt'

flags = tf.flags

# path related
flags.DEFINE_string('load_model',           None,           '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')


# model params
flags.DEFINE_integer('word_vocab_size',     1738,                               'size of vocabulary')
flags.DEFINE_integer('rnn_size',            128,                                'size of LSTM cell')
flags.DEFINE_integer('highway_layers',      1,                                  'size of highway layers')
flags.DEFINE_integer('word_embed_size',     72,                                'embed_size')
flags.DEFINE_string ('kernels',              '[1,2,3,4,5,6,7]',                  'CNN kernel width')
flags.DEFINE_string ('kernel_features',     '[25,50,75,100,125,150,175]',       'CNN kernel num')
flags.DEFINE_integer('rnn_layers',          1,                                  'num of layers of RNN')
flags.DEFINE_float  ('dropout',             0.0,                                'dropout')

# optimization

flags.DEFINE_float  ('learning_rate_decay', 0.5,               'learning rate decay')
flags.DEFINE_float  ('learning_rate',       0.002,             'lr')
flags.DEFINE_float  ('decay_when',          1.0,               'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    120,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          32,   'number of sequences to train on in parallel')
flags.DEFINE_integer('batch_size_eval',     3,    'number of sequences to evaluate in parallel at eval time')
flags.DEFINE_integer('max_epochs',          100,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_sent_length',     30,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435,    'random number generator seed')
flags.DEFINE_integer('print_every',    5,       'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',     '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
flags.DEFINE_integer('patience',       10000,   'tolerance for early_stopping')

FLAGS = flags.FLAGS



def train():


    dataset_tensors, labels_tensors = dl.make_batches()

    input_tensor_tr, label_tensor_tr, seq_tensor_tr = dl.sequence_init(dataset_tensors, labels_tensors, FLAGS.num_unroll_steps, 'Train', allow_short_seq= False)
    input_tensor_te, label_tensor_te, seq_tensor_te = dl.sequence_init(dataset_tensors, labels_tensors, FLAGS.num_unroll_steps, 'Test', allow_short_seq= True)

    train_reader = dl.TrainDataReader(input_tensor_tr, label_tensor_tr, seq_tensor_tr, FLAGS.batch_size, FLAGS.num_unroll_steps, False)
    eval_reader = dl.EvalDataReader(input_tensor_te, label_tensor_te, seq_tensor_te, FLAGS.batch_size_eval, FLAGS.num_unroll_steps, False)

    '''

    input_tensors, label_tensors, seq_tensors = dl.make_batches(60)
    train_reader = dl.DataReader(input_tensors['Train'], label_tensors['Train'],
                                 seq_tensors['Train'], FLAGS.batch_size, FLAGS.num_unroll_steps)

    eval_reader = dl.DataReader(input_tensors['Devel'], label_tensors['Devel'], seq_tensors['Devel'],
                                FLAGS.batch_size, FLAGS.num_unroll_steps)
    '''

    labels = tf.placeholder(tf.float32, [None, FLAGS.num_unroll_steps, 3], name = 'labels')

    #labels = tf.reshape(labels, [-1, 3])

    train_model = model.inference_graph(word_vocab_size= FLAGS.word_vocab_size,
                                        kernels= eval(FLAGS.kernels),
                                        kernel_features= eval(FLAGS.kernel_features),
                                        rnn_size= FLAGS.rnn_size,
                                        dropout= FLAGS.dropout,
                                        num_rnn_layers= FLAGS.rnn_layers,
                                        num_highway_layers= FLAGS.highway_layers,
                                        num_unroll_steps= FLAGS.num_unroll_steps,
                                        max_sent_length= FLAGS.max_sent_length,
                                        #batch_size= FLAGS.batch_size,
                                        embed_size= FLAGS.word_embed_size)

    predictions = train_model.predictions

    #print(predictions)


    losses = model.loss_graph(predictions, labels)

    eval_model = model.eval_metric_graph()

    loss_arousal = losses.loss_arousal
    loss_valence = losses.loss_valence
    loss_liking = losses.loss_liking

    #loss_list = [(model.loss_graph(predictions[:,i], labels[:,i]) for i in range(3))]

    #print(loss_list)
    #loss = tf.convert_to_tensor(loss_list)

    #metric = [1. - x for x in loss_list]

    metric_arousal = 1. - loss_arousal
    metric_valence = 1. - loss_valence
    metric_liking = 1. - loss_liking

    eval_arousal = eval_model.eval_metric_arousal
    eval_valence = eval_model.eval_metric_valence
    eval_liking = eval_model.eval_metric_liking

    loss_op = loss_arousal + loss_liking + loss_valence

    optimizer = tf.train.AdamOptimizer(learning_rate= FLAGS.learning_rate).minimize(loss_op)

    saver = tf.train.Saver()

    patience = FLAGS.patience

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        best_metric_arousal = 0.0
        best_metric_valence = 0.0
        best_metric_liking = 0.0


        Done = False

        epoch = 0

        while epoch < FLAGS.max_epochs and not Done:

            batch = 1
            epoch += 1

            for minibatch in train_reader.iter():

                x, y = minibatch

                #print(x.shape, y.shape)

                _, l, m_arousal, m_valence, m_liking = sess.run(
                    [optimizer, loss_op, metric_arousal, metric_valence, metric_liking],
                    feed_dict={
                    train_model.input: x,
                    labels: y,
                    train_model.sequence_length: [120] * FLAGS.batch_size,
                    train_model.batch_size: FLAGS.batch_size
                    })

                print('Epoch: %5d/%5d -- batch: %5d -- loss: %.4f' % (epoch, FLAGS.max_epochs, batch, l))

                if batch % 3 == 0:
                    print('arousal: %.4f -- valence: %.4f, liking: %.4f'
                          % (m_arousal, m_valence, m_liking))
                    log = open(LOGGING_PATH, 'a')
                    log.write('%s, %6d, %.5f, %.5f, %.5f, %.5f, \n' % ('train', epoch * batch,
                                                                       l, m_arousal, m_valence, m_liking))
                    log.close()


                if batch % 14 == 0:
                    print('evaluation process------------------------------------------')

                    eval_metric = []
                    cnt = 0
                    prev = None
                    for mb in eval_reader.iter():

                        eval_x_list, eval_y_list, eval_z_list = mb

                        for eval_x, eval_z in zip(eval_x_list, eval_z_list):
                            cnt += np.sum(eval_z)
                            eval_tmp_preds = sess.run([predictions], feed_dict={
                                train_model.input : eval_x,
                                train_model.sequence_length : eval_z,
                                train_model.batch_size: FLAGS.batch_size_eval
                            })

                            if prev is None: prev = eval_tmp_preds[0]
                            else: prev = np.vstack((prev, eval_tmp_preds[0]))
                        prev = prev[:cnt]
                        eval_y_list = np.array(eval_y_list).reshape([-1, 3])[:cnt]

                        #print(prev)
                        #print(eval_y_list)

                        e_arousal, e_valence, e_liking = sess.run([eval_arousal, eval_liking, eval_valence],
                                                        feed_dict= {
                                                            eval_model.eval_predictions : prev,
                                                            eval_model.eval_labels : eval_y_list
                                                        })


                        eval_metric.append([e_arousal, e_valence, e_liking])
                        prev = None
                        cnt = 0

                    eval_res = np.mean(np.array(eval_metric), axis= 0)
                    eval_loss = np.sum(1. - eval_res)
                    print('Epoch: %5d/%5d -- batch: %5d -- loss: %.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                          % (epoch, FLAGS.max_epochs, batch, eval_loss, eval_res[0], eval_res[1], eval_res[2]))

                    log = open(LOGGING_PATH, 'a')
                    log.write('%s, %6d, %.5f, %.5f, %.5f, %.5f, \n' % ('train',
                                                                       epoch * batch, eval_loss, eval_res[0],
                                                                       eval_res[1], eval_res[2]))
                    log.close()
                    print('done evaluation------------------------------------------\n')

                '''
                if batch % 10 == 0:

                    print('evaluation process------------------------------------------')
                    metr = []
                    eval_loss = 0.0
                    cnt = 0

                    for mb in eval_reader.iter():
                        eval_x, eval_y = mb
                        cnt += 1

                        l_e, me_arousal, me_valence, me_liking = sess.run(
                            [loss_op, metric_arousal, metric_valence, metric_liking], feed_dict={
                            train_model.input: eval_x,
                            labels: eval_y
                        })

                        eval_loss += l_e

                        metr.append([me_arousal, m_valence, me_liking])

                    mean_metr = np.mean(np.array(metr), axis= 0)
                    eval_loss /= cnt

                    if mean_metr[0] > best_metric_arousal or mean_metr[1] > best_metric_valence \
                            or mean_metr[2] > best_metric_liking:
                        save_path = saver.save(sess, SAVE_PATH)

                        best_metric_arousal, best_metric_valence, best_metric_liking = mean_metr[0], \
                                                        mean_metr[1], mean_metr[2]
                        patience = FLAGS.patience
                        print('Model saved in file: %s' % save_path)

                    else:
                        patience -= 500
                        patience -= 500
                        if patience <= 0:
                            Done = True
                            break

                    print('Epoch: %5d/%5d -- batch: %5d -- loss: %.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                          % (epoch, FLAGS.max_epochs, batch, eval_loss, mean_metr[0], mean_metr[1], mean_metr[2]))

                    log = open(LOGGING_PATH, 'a')
                    log.write('%s, %6d, %.5f, %.5f, %.5f, %.5f, \n' % ('train',
                                                epoch * batch, eval_loss, mean_metr[0], mean_metr[1], mean_metr[2]))
                    log.close()
                    print('done evaluation------------------------------------------\n')
                '''
                batch += 1



if __name__ == '__main__':
    train()



