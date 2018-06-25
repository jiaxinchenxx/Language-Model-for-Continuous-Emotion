import numpy as np
import pickle
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import collections


Transcription_PATH = 'D:\AVEC2017_SEWA\\transcriptions'
Labels_PATH = 'D:\AVEC2017_SEWA\labels'
Save_PATH = '.\data_copy'

train_set = [i for i in range(1, 35)]
test_set = [i for i in range(1, 17)]
dev_set = [i for i in range(1, 15)]
inx_dic = {'Train_set' : train_set, 'Test_set' : test_set, 'Devel_set' : dev_set}


class Vocab(object):

    def __init__(self, token2index = None, index2token = None):

        self._token2index = token2index or {}
        self._index2token = index2token or []


    def feed(self, token):
        if token not in self._token2index:

            index = len(self._index2token)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]


    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            return KeyError(token)
        return index


    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        filename = os.path.join(Save_PATH, filename)
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f)

    def load(cls, filename):
        filename = os.path.join(Save_PATH, filename)
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return token2index, index2token


def savePickle(file, filename):
    path = os.path.join(Save_PATH, filename)
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def loadPickle(filename):
    path = os.path.join(Save_PATH, filename)
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

class turnsInfo(object):
    def __init__(self, timestep, text2inx):
        self.timestep = timestep
        self.text2inx = text2inx



def load_corpus(data_dir = None, max_sentence_length = None):

    word_vocab = Vocab()
    word_vocab.feed('<unk>') # this is for the non-update padding, which is just <unk> word, the word_embedding won't update
                             # at the index [0] further
    max_sent_length = 0
    tokenizer = RegexpTokenizer(r'\w+')

    dataset_dic = collections.defaultdict(list)
    labels_dic = collections.defaultdict(list)

    for fname in ['Train', 'Devel', 'Test']:
        print('reading ' + fname + ' file')
        for inx in inx_dic[fname + '_set']:
            filename = fname + '_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.csv'
            print('reading file: ' + filename)
            with open(os.path.join(Transcription_PATH, filename), 'r', encoding= 'utf-8') as f:
                lines = f.readlines()

            with open(os.path.join(Labels_PATH, filename), 'r', encoding= 'utf-8') as f:
                tmp = f.readlines()



            frames = [[] for _ in range(len(tmp))]
            labels = []

            for line in tmp:  # gather labels for all the time steps for one file(one subject)
                label = line.split(';')
                label = [float(label[2]), float(label[3]), float(label[4])]
                labels.append(label)

            for line in lines:  # fill the frame within a turn duration with the index of words of that duration
                separate_parts = line.split(';')
                text = separate_parts[2]
                text = tokenizer.tokenize(text)
                inx_array = [word_vocab.feed(word.lower()) for word in text]

                st = int(round(float(separate_parts[0]), 1) * 10)  # calculate the duration of the turns
                end = int(round(float(separate_parts[1]), 1) * 10)

                for i in range(st, end + 1 if end + 1 < len(tmp) else len(tmp)): frames[i] = inx_array
                max_sent_length = max(max_sent_length, len(inx_array))

            #for i in range(len(tmp)):
            #    dataset_dic[filename].append(turnsInfo(float(i / 10), frames[i]))
            #frames.append(turnsInfo(float(separate_parts[0]), float(separate_parts[1]), inx_array))

            dataset_dic[filename] = frames # record the frames info of certain file(subject)
            labels_dic[filename] = labels # as well as their labels

            assert len(dataset_dic[filename]) == len(labels_dic[filename])
            #dataset_dic[filename]['length'] = max_sentence_length
            #dataset_dic[filename]['value'] = frames

    word_vocab.save('vocabulary.pkl')
    savePickle(dataset_dic, 'dataset_dic_without_timestep.pkl')
    savePickle(labels_dic, 'labels_dic.pkl')
    return dataset_dic, labels_dic, max_sent_length

def createTensor(dataset_dic, labels_dic, max_sent_length):  # generate np.array form of the input_ and labels
                                                             # with shape = [len(input_), max_sent_length]
    dataset_tensors = {}
    labels_tensors = {}
    for fname in ['Train', 'Devel', 'Test']:
        print('processing ' + fname + ' file')
        for inx in inx_dic[fname + '_set']:
            filename = fname + '_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.csv'

            data = dataset_dic[filename]
            label = labels_dic[filename]

            frames = np.zeros([len(data), max_sent_length], dtype= np.int32) # feed the frame without any word
            labels = np.zeros([len(data), 3], dtype= np.float32)            # with 0,

            for i, word_array in enumerate(data):
                frames[i, :len(word_array)] = np.array(word_array) # feed the frame within the duration of turns
                labels[i] = label[i]                        # with word index of that duration
            dataset_tensors[filename] = frames  # record the info of words of each frame of each file(subject)
            labels_tensors[filename] = labels

    savePickle(dataset_tensors, 'dataset_tensors.pkl')  # contains tensors with shape = [len, max_sent_length]
    savePickle(labels_tensors, 'labels_tensors.pkl')  # contains tensors with shape = [len, 3]
    return dataset_tensors, labels_tensors


def sequence_init(dataset_tensors, labels_tensors, num_unroll_steps, isinclude_last = False, isShuffle = False):
    '''
    here allow_short_seq is FALSE, means we will automatically discard the final frames that cannot form a valid
    sequence, however, if it is TRUE, then we will allow the shorter frames,
    which may require further operation on the seq_length_tensors,
    give seq_length a TF.PALCEHOLDER, to enable the DYNAMIC_RNN to know where to stop for shorter seq_lenth frames,
    also for us to know how long we need to calculate the loss or metric

    :param dataset_tensors: includes dataset_tensors['Test/Train/Devel)_01.csv'] .. dataset_tensors['Test/Train/Devel)_16.csv'] ..
    :param labels_tensors: includes labels_tensors['Test/Train/Devel)_01.csv'] .. labels_tensors['Test/Train/Devel)_16.csv'] ..
    :param num_unroll_steps: how many time steps in a sequence
    :param isDiscard: whether or not to discard the final time steps that cannot form a sequence of num_unroll_steps size
    :param isShuffle: whether or not to shuffle all the sequences
    :return: ndarray form of input_[Train/Test/Devel] and label_[Train/Test/Devel] shape = [# of sequences, num_unroll_steps, max_sent_length]
                [# of sequences, num_unroll_steps, 3]
    '''


    input_tensors = collections.defaultdict(list)
    label_tensors = collections.defaultdict(list)
    seqlen_tensors = collections.defaultdict(list)

    allow_short_seq = False
    for fname in ['Train', 'Devel', 'Test']:
        #allow_short_seq = False if fname == 'Train' else True
        for inx in inx_dic[fname + '_set']:
            filename = fname + '_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.csv'
            print('processing ' + filename)


            frames, labels, sequence_length_list = make_sequence(num_unroll_steps, dataset_tensors[filename],
                                           labels_tensors[filename], isinclude_last, allow_short_seq)

            # make the frames into several sequences of one subject from Train/Test/Devel

            input_tensors[fname].extend(frames) # add these sequences into their list Train/Test/Devel
            label_tensors[fname].extend(labels)
            seqlen_tensors[fname].extend(sequence_length_list)

            assert len(input_tensors[fname]) == len(seqlen_tensors[fname])


    for fname in ['Train', 'Devel', 'Test']:
        input_tensors[fname] = np.array(input_tensors[fname]) # make it type as ndarray
        label_tensors[fname] = np.array(label_tensors[fname])
        seqlen_tensors[fname] = np.array(seqlen_tensors[fname])
        #print(label_tensors[fname][0])


        length = len(input_tensors[fname]) # length of Train/Test/Devel sequences, how many sequences are there

        if isShuffle:
            inx = np.arange(length)
            np.random.shuffle(inx)
            input_tensors[fname] = input_tensors[fname][inx]
            label_tensors[fname] = label_tensors[fname][inx]
            seqlen_tensors[fname] = seqlen_tensors[fname][inx]
        #print(input_tensors[fname].shape)

    savePickle(input_tensors, 'final_input_tensors.pkl')
    savePickle(label_tensors, 'final_label_tensors.pkl')
    savePickle(seqlen_tensors, 'final_seqlen_tensors.pkl')

    return input_tensors, label_tensors, seqlen_tensors



def make_sequence(num_unroll_steps, x, y, isinclude_last = False, allow_short_seq = False):
    assert len(x) == len(y)
    assert len(x.shape) == 2

    frames = []
    labels = []

    n = len(x)
    num_sequences = n // num_unroll_steps #  how many sequences could be made

    for i in range(num_sequences):
        fr_seq = x[i * num_unroll_steps: (i + 1) * num_unroll_steps]
        la_seq = y[i * num_unroll_steps: (i + 1) * num_unroll_steps]
        frames.append(fr_seq)
        labels.append(la_seq)


    if n % num_unroll_steps != 0 and isinclude_last: # if we don't want to discard the last several frames when training
        st = n - num_sequences * num_unroll_steps    # we want them to be in a full sequence
        for i in range(num_sequences):
            fr_seq = x[st + i * num_unroll_steps: (i + 1) * num_unroll_steps + st]
            la_seq = y[st + i * num_unroll_steps: (i + 1) * num_unroll_steps + st]
            frames.append(fr_seq)
            labels.append(la_seq)

        assert len(frames) == len(labels)
        return frames, labels, [num_unroll_steps] * len(frames)

    assert len(frames) == len(labels)

    if not isinclude_last and not allow_short_seq: # discard the last frams that cannot form a sequence for training set
        return frames, labels, [num_unroll_steps] * len(frames)
    elif allow_short_seq:

        fr_seq = np.zeros([num_unroll_steps, x.shape[1]], dtype= np.int32)
        la_seq = np.zeros([num_unroll_steps, 3], dtype= np.float32)

        fr_seq[:n - num_sequences * num_unroll_steps] = x[num_sequences * num_unroll_steps:]
        la_seq[:n - num_sequences * num_unroll_steps] = y[num_sequences * num_unroll_steps:]

        frames.append(fr_seq)
        labels.append(la_seq)

        return frames, labels, [num_unroll_steps] * num_sequences + [n - num_unroll_steps * num_sequences]

def make_batches(num_unroll_steps, max_sent_length = 30):
    dataset_dic = loadPickle('dataset_dic_without_timestep.pkl')
    labels_dic = loadPickle('labels_dic.pkl')

    vocab = Vocab()
    a, b = vocab.load('vocabulary.pkl')
    print(len(b), b)

    dataset_tensors, labels_tensors = createTensor(dataset_dic, labels_dic, max_sent_length)

    input_tensors, label_tensors, seqlen_tensors = sequence_init(dataset_tensors, labels_tensors, num_unroll_steps)

    return input_tensors, label_tensors, seqlen_tensors

class DataReader(object):

    def __init__(self, dataset_tensors, labels_tensors, seqlen_tensors, batch_size, num_unroll_steps, isReducedLength = True):
        '''
        here we just ignore the sequences which couldn't form a batch, which leads to the reduced_length,
        this is better for us to yield batches
        however, could also use original length
        :param dataset_tensors:
        :param labels_tensors:
        :param seqlen_tensors:
        :param batch_size:
        :param num_unroll_steps:
        :param isReducedLength:
        '''
        length = len(dataset_tensors)
        print(length)
        assert length == len(labels_tensors)
        assert length == len(seqlen_tensors)

        max_sent_length = dataset_tensors.shape[2]

        reduced_length = (length // batch_size) * batch_size
        dataset_tensors = dataset_tensors[:reduced_length]
        labels_tensors = labels_tensors[:reduced_length]

        #seqlen_tensors = seqlen_tensors[:reduced_length]
        #print(seqlen_tensors)

        #print(type(dataset_tensors))
        #print(len(dataset_tensors))
        #print(labels_tensors)

        '''original work use transpose below to implement shuffles'''
        x_batches = dataset_tensors.reshape([-1, batch_size, num_unroll_steps, max_sent_length])
        y_batches = labels_tensors.reshape([-1, batch_size, num_unroll_steps, 3])

        #seq_batches = seqlen_tensors.reshape([-1, batch_size])

        #x_batches = np.transpose(x_batches, axes = [1, 0, 2, 3])
        #y_batches = np.transpose(y_batches, axes = [1, 0, 2, 3])
        #print(y_batches[0].shape)
        #print(seq_batches)

        print(x_batches.shape)
        print(y_batches.shape)
        #print(seq_batches.shape)

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        #self._seq_batches = list(seq_batches)

        assert len(self._x_batches) == len(self._y_batches)

        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y

if __name__ == '__main__':

    #dataset_dic, labels_dic, max_sent_length = load_corpus()

    dataset_dic = loadPickle('dataset_dic_without_timestep.pkl')
    labels_dic = loadPickle('labels_dic.pkl')

    vocab = Vocab()
    a, b = vocab.load('vocabulary.pkl')
    print(len(b), b)

    #dic = loadPickle('dataset_dic_without_timestep.pkl')
    for item in dataset_dic['Test_01.csv']:
        print(item)
    for item in labels_dic['Test_01.csv']:
        print(item)

    dataset_tensors, labels_tensors = createTensor(dataset_dic, labels_dic, 30)

    input_tensors, label_tensors, seqlen_tensors = sequence_init(dataset_tensors, labels_tensors, 60)

    train_reader = DataReader(input_tensors['Test'], label_tensors['Test'], seqlen_tensors['Test'], 32, 60, False)

    #for x, y in train_reader.iter():
    #    print(x.shape, y.shape)

    #print(dataset_tensors['Test_01.csv'])
    #print(labels_tensors['Test_01.csv'])


"""
p = '... ich hätte ihn gleich fast wieder vergessen, diesen Werbespot'
tokenizer = RegexpTokenizer(r'\w+')
output = tokenizer.tokenize(p)
print(output)
stops = stopwords.words('german')
output = [word for word in output if not word in stops]
print (output)
"""


