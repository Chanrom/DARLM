#coding=utf-8
import codecs
import torch
import numpy as np
import cPickle
from utils import constant
import time

np.random.seed(188)

class PreProcess(object):
    """
    preprosess for sick data
    """
    def __init__(self, train_src='data/sst_data/raw_data/sst_train.txt', valid_src='data/sst_data/raw_data/sst_dev.txt',
                 test_src='data/sst_data/raw_data/sst_test.txt', corpus_src='data/sst_data/raw_data/sst_all.txt',
                 vocab_size=25000, binary=True, shuffle=False, sorted_size=False, seed=1234, lower=False,
                 save_data='data/sst_data/raw_data/'
                ):
        super(PreProcess, self).__init__()

        self.train_src = train_src
        self.valid_src = valid_src
        self.test_src = test_src
        self.corpus_src = corpus_src
        self.vocab_size = vocab_size
        self.binary = binary
        self.shuffle = shuffle
        self.sorted_size = sorted_size
        self.lower = lower
        self.save_data = save_data

        self.id2token = {}
        self.token2id = {}
        self.frequences = {}
        
        torch.manual_seed(seed)

    def makeVocab(self, filepath):
        frequences = {}
        lines = codecs.open(filepath, 'r', 'utf-8').readlines()
        max_len = 0
        for line in lines:

            # for SST data
            l = line.strip().split('\t')
            words = l[1].split(' ')

            if len(words) > max_len:
                max_len = len(words)

            for word in words:
                if self.lower:
                    word = word.lower()
                if not frequences.has_key(word):
                    frequences[word] = 1
                else:
                    frequences[word] = frequences[word] + 1

        self.token_index = 0
        if len(constant.special_tokens) > 0:
            # 扣去两个指标给特殊字符
            self.vocab_size = self.vocab_size - len(constant.special_tokens)
            for item in constant.special_tokens:
                self.token2id[item] = self.token_index
                self.id2token[self.token_index] = item
                self.token_index += 1
                self.frequences[item] = 1

        freq = sorted(frequences.items(), key=lambda d: d[1], reverse=True)
        vocab_size = len(freq) if self.vocab_size > len(freq) else self.vocab_size
        for i in range(vocab_size):
            word = freq[i][0]
            count = freq[i][1]

            self.token2id[word] = self.token_index
            self.id2token[self.token_index] = word
            self.token_index += 1

            self.frequences[word] = count

        print ' ', 'RAW Max len', max_len
        print ' ', 'RAW Max vocabulary size', len(self.token2id)

        # cPickle.dump(self.token2id.keys(),open('data/vocab.pkl', 'wb'))

        return {'size':len(self.token2id), 'token2id':self.token2id, 'id2token':self.id2token, 'freq':self.frequences}

    def makeData(self, filepath):
        lines = codecs.open(filepath, 'r', 'utf-8').readlines()
        s_data = []
        l_data = []
        sizes = []
        _i = 0
        unique_sent = {}
        for line in lines:
            _i += 1
            # print _i
            line = line.lower() if self.lower else line
            l = line.strip().split('\t')

            sentence = l[1]
            tag = int(l[0])

            if self.binary:
                if tag > 2:
                    label = 1
                elif tag < 2:
                    label = 0
                else:
                    continue
            else:
                label = tag

            if not unique_sent.has_key(sentence):
                words = sentence.split(' ')
                s_data.append(torch.LongTensor([self.token2id[word] if self.token2id.has_key(word) else constant.UNK_ID for word in words]))
                l_data.append(torch.LongTensor([label]))
                sizes.append(len(words))

        if self.shuffle:
            print '  Shuffling sentence'
            perm = torch.randperm(len(s_data))
            s_data = [s_data[idx] for idx in perm]
            l_data = [l_data[idx] for idx in perm]
            sizes = [sizes[idx] for idx in perm]

        if self.sorted_size:
            print '  Sorting sentence by size'
            _, perm = torch.sort(torch.Tensor(sizes))
            s_data = [s_data[idx] for idx in perm]
            l_data = [l_data[idx] for idx in perm]

        print '  Got total', len(s_data), 'samples'

        # print '---', s_data[0]

        return s_data, l_data

    def wordvec(self, flag):
        lines = codecs.open('data/sst_data/raw_data/sst_glove_300d.txt', 'r', 'utf-8').readlines()
        wordvec = {}
        for line in lines:
            l = line.strip().split()
            word = l[0]
            vec = [float(i) for i in l[1:]]
            wordvec[word] = vec

        wv = np.zeros((len(self.token2id), 300))
        count = 0
        for i in range(len(self.token2id)):
            word = self.id2token[i]
            if wordvec.has_key(word):
                count += 1
            vec = wordvec[word] if wordvec.has_key(word) else np.random.uniform(-0.1, 0.1, size=(1, 300))
            wv[i] = vec
        print ' ', count, 'found in Glove'
        file_name = self.save_data + 'wv_noex2_%s.npz'%flag if self.binary else self.save_data + 'wv_noex5_%s.npz'%flag
        print ' Save to', file_name
        wv.dump(file_name)


    def process(self):

        time_stamp = time.strftime("%Y_%m_%d", time.localtime()) 

        print '* Build vocabulary...'
        vocab = self.makeVocab(self.corpus_src)
        print '* Bulid OK.'

        print '* Prepare training data...'
        train = self.makeData(self.train_src)

        print '  Prepare valid data...'
        valid = self.makeData(self.valid_src)

        print '  Prepare test data...'
        test = self.makeData(self.test_src)
        print '* Prepare OK.'

        outfile = 'train_noex2_%s.pt'%time_stamp if self.binary else 'train_noex5_%s.pt'%time_stamp
        print '* Saving data to \'' + self.save_data + outfile + '...'
        save_data = {'vocab': vocab,
                     'train': train,
                     'valid': valid,
                     'test': test}
        torch.save(save_data, self.save_data + outfile)
        print '* OK.'

        # get pre-trained wordvec
        print '* Dump pre-trained wordvec...'
        self.wordvec(time_stamp)
        print '* OK.'

p = PreProcess(shuffle=True, binary=False)
p.process()

