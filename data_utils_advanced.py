import numpy as np
import pickle
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import collections
#np.set_printoptions(threshold= np.nan)

Transcription_PATH = 'D:\AVEC2017_SEWA\\transcriptions'
Labels_PATH = 'D:\AVEC2017_SEWA\labels'
Save_PATH = '.\data_copy_2'

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

            frames = np.zeros([len(data), max_sent_length], dtype=np.int32)  # feed the frame without any word
            labels = np.zeros([len(data), 3], dtype=np.float32)  # with 0,

            for i, word_array in enumerate(data):
                frames[i, :len(word_array)] = np.array(word_array)  # feed the frame within the duration of turns
                labels[i] = label[i]  # with word index of that duration
            dataset_tensors[filename] = frames  # record the info of words of each frame of each file(subject)
            labels_tensors[filename] = labels

    savePickle(dataset_tensors, 'dataset_tensors.pkl')  # contains tensors with shape = [len, max_sent_length]
    savePickle(labels_tensors, 'labels_tensors.pkl')  # contains tensors with shape = [len, 3]
    return dataset_tensors, labels_tensors



def sequence_init(dataset_tensors, labels_tensors, num_unroll_steps, fname, allow_short_seq = False, isShuffle = False):

    input_tensor = []
    label_tensor = []
    seq_tensor   = []

    for inx in inx_dic[fname + '_set']:
        filename = fname + '_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.csv'
        print('processing ' + filename)

        frames, labels, sequence_length_list = make_sequence(num_unroll_steps, dataset_tensors[filename],
                                                             labels_tensors[filename], allow_short_seq)

        if fname == 'Train':
            input_tensor.extend(frames)
            label_tensor.extend(labels)
            seq_tensor.extend(sequence_length_list)

        else:
            input_tensor.append(np.array(frames))
            label_tensor.append(np.array(labels))
            seq_tensor.append(np.array(sequence_length_list))

    if fname == 'Train':
        input_tensor = np.array(input_tensor)
        label_tensor = np.array(label_tensor)
        seq_tensor = np.array(seq_tensor)

    savePickle(input_tensor, 'input_tensors_%s.pkl' % fname)
    savePickle(label_tensor, 'label_tensors_%s.pkl' % fname)
    savePickle(seq_tensor, 'seq_tensors_%s.pkl' % fname)

    return input_tensor, label_tensor, seq_tensor

def make_sequence(num_unroll_steps, x, y, allow_short_seq = False):
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

    assert len(frames) == len(labels)

    if not allow_short_seq: # discard the last frames that cannot form a sequence for training set
        return frames, labels, [num_unroll_steps] * len(frames)
    else:
        fr_seq = np.zeros([num_unroll_steps, x.shape[1]], dtype= np.int32)
        la_seq = np.zeros([num_unroll_steps, 3], dtype= np.float32)

        fr_seq[:n - num_sequences * num_unroll_steps] = x[num_sequences * num_unroll_steps:]
        la_seq[:n - num_sequences * num_unroll_steps] = y[num_sequences * num_unroll_steps:]

        frames.append(fr_seq)
        labels.append(la_seq)

        return frames, labels, [num_unroll_steps] * num_sequences + [n - num_unroll_steps * num_sequences]


def make_sequence_with_delay(num_unroll_steps, x, y, allow_short_seq = False, delay=10):
    assert len(x) == len(y)
    assert len(x.shape) == 2

    frames = []
    labels = []
    n = len(x)

    num_sequences = (n - delay) // num_unroll_steps

    for i in range(num_sequences):  # this function is for time delay
        fr_seq = x[i * num_unroll_steps: (i + 1) * num_unroll_steps]
        la_seq = y[i * num_unroll_steps + delay: (i + 1) * num_unroll_steps + delay]
        frames.append(fr_seq)
        labels.append(la_seq)

def make_batches():
    # data_dic, labels_dic contain the raw word inx and labels of every frame of every file

    dataset_dic, labels_dic, max_sent_length = load_corpus()

    # dataset_tensors, labels_tensors contain the structured wordinx and labels as ndarray form of each file
    # which means, each key(file) contains an ndarray of shape = [file_length, max_sent_length] labels shape = [file_length, 3]

    dataset_tensors, labels_tensors = createTensor(dataset_dic, labels_dic, max_sent_length)

    return dataset_tensors, labels_tensors


class TrainDataReader(object):

    def __init__(self, input_tensor, label_tensor, seq_tensor, batch_size, num_unroll_steps, isReducedLength = True):

        length = len(input_tensor)
        print(input_tensor.shape)
        assert length == len(label_tensor)
        assert length == len(seq_tensor)

        max_sent_length = input_tensor.shape[2]

        if isReducedLength:
            reduced_length = (length // batch_size) * batch_size
            input_tensor = input_tensor[:reduced_length]
            label_tensor = label_tensor[:reduced_length]
            seq_tensor = seq_tensor[:reduced_length]

        self.x_batches = []
        self.y_batches = []
        self.seq_batches = []

        num_batches = length // batch_size

        for i in range(num_batches):
            self.x_batches.append(input_tensor[i * batch_size : (i + 1) * batch_size])
            self.y_batches.append(label_tensor[i * batch_size : (i + 1) * batch_size])
            self.seq_batches.append(seq_tensor[i * batch_size : (i + 1) * batch_size])

        if length % batch_size != 0:
            self.x_batches.append(input_tensor[num_batches * batch_size:])
            self.y_batches.append(label_tensor[num_batches * batch_size:])
            self.seq_batches.append(seq_tensor[num_batches * batch_size:])

        self.length = len(self.y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):
        for x, y in zip(self.x_batches, self.y_batches):
            yield x, y


class EvalDataReader(object):

    def __init__(self, input_tensor, label_tensor, seq_tensor, batch_size, num_unroll_steps, isReducedLength=True):

        length = len(input_tensor)
        print(input_tensor[0].shape)
        assert length == len(label_tensor)
        assert length == len(seq_tensor)

        self.x_batches = []
        self.y_batches = []
        self.seq_batches = []


        length = len(input_tensor[0])
        assert length % batch_size == 0
        cnt = 0

        for input, label, seqlen in zip(input_tensor, label_tensor ,seq_tensor):
            self.x_batches.append([])
            self.y_batches.append(label)
            self.seq_batches.append([])
            for i in range(length // batch_size):
                self.x_batches[cnt].append(input[i * batch_size: (i + 1) * batch_size])
                #self.y_batches[cnt].append(label[i * batch_size: (i + 1) * batch_size])
                self.seq_batches[cnt].append(seqlen[i * batch_size: (i + 1) * batch_size])
            cnt += 1

        self.length = len(self.y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):

        for x, y, z in zip(self.x_batches, self.y_batches, self.seq_batches):
            yield x, y, z



'''
#data_dic, labels_dic contain the raw word inx and labels of every frame of every file

dataset_dic, labels_dic, max_sent_length = load_corpus()

#dataset_tensors, labels_tensors contain the structured wordinx and labels as ndarray form of each file
# which means, each key(file) contains an ndarray of shape = [file_length, max_sent_length] labels shape = [file_length, 3]
dataset_tensors, labels_tensors = createTensor(dataset_dic, labels_dic, max_sent_length)


# input_tensor gathers all the information of Train/Test/Devel
# if Train, then it will be a large tensor, which is formed by first cut each train_file into num_unroll_steps sequences,
# which doesn't allow shorter sequence,
# then concatenate all the sequecnes from one file together,
# shape = [# of sequences of all train file, num_unroll_steps, max_sent_length]
input_tensor_tr, label_tensor_tr, seq_tensor_tr = sequence_init(dataset_tensors, labels_tensors, 60, 'Train', allow_short_seq= False)
input_tensor_te, label_tensor_te, seq_tensor_te = sequence_init(dataset_tensors, labels_tensors, 60, 'Test', allow_short_seq= True)

train_reader = TrainDataReader(input_tensor_tr, label_tensor_tr, seq_tensor_tr, batch_size= 32, num_unroll_steps= 60, isReducedLength= False)
eval_reader = EvalDataReader(input_tensor_te, label_tensor_te, seq_tensor_te, batch_size= 3, num_unroll_steps= 60, isReducedLength= False)

for x, y, z in eval_reader.iter():
    print(x, y, z)
'''