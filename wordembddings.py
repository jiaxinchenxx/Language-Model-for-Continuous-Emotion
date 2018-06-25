import numpy as np
import pickle
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import collections

Transcription_PATH = 'D:\AVEC2017_SEWA\\transcriptions'
Labels_PATH = 'D:\AVEC2017_SEWA\labels'
Save_PATH = '.\word_embedding'

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


def load_corpus(data_dir = None, max_sentence_length = None):

    word_vocab_train = Vocab()
    word_vocab_train.feed('<unk>') # this is for the non-update padding, which is just <unk> word, the word_embedding won't update
                             # at the index [0] further

    word_vocab_dev = Vocab()
    word_vocab_dev.feed('<unk>')

    word_vocab_test = Vocab()
    word_vocab_test.feed('<unk>')


    word_vocab = {'Train' : word_vocab_train, 'Devel' : word_vocab_dev, 'Test' : word_vocab_test}

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
                inx_array = [word_vocab[fname].feed(word.lower()) for word in text]

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

        word_vocab[fname].save('vocabulary_{}.pkl'.format(fname))
    savePickle(dataset_dic, 'dataset_dic_without_timestep.pkl')
    savePickle(labels_dic, 'labels_dic.pkl')
    return word_vocab, dataset_dic, labels_dic, max_sent_length

vocab_test = Vocab()
vocab_train = Vocab()

token2inx_test, _ = vocab_test.load('vocabulary_Test.pkl')
token2inx_train, _ = vocab_train.load('vocabulary_Train.pkl')

for key in token2inx_train.keys():
    print(key)
print(len(token2inx_train.keys()),'-----------------------------------')

for key in token2inx_test.keys():
    if key not in token2inx_train.keys():
        print(key)
print(len(token2inx_test.keys()))

wordembeddings = gensim.models.KeyedVectors.load_word2vec_format('german.model', binary= True)