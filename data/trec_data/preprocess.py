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
    preprosess for trec data
    """
    def __init__(self, train_src='data/trec_data/raw_data/trec_split_train.txt', valid_src='data/trec_data/raw_data/trec_split_valid.txt',
                 test_src='data/trec_data/raw_data/trec_test.txt', corpus_src='data/trec_data/raw_data/trec_all.txt',
                 vocab_size=25000, shuffle=False, sorted_size=False, seed=1234, lower=False,
                 save_data='data/trec_data/raw_data/'
                ):
        super(PreProcess, self).__init__()

        self.train_src = train_src
        self.valid_src = valid_src
        self.test_src = test_src
        self.corpus_src = corpus_src
        self.vocab_size = vocab_size
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
        tags = {}
        for line in lines:

            # for TREC data
            l = line.strip().split(' ')
            words = l[1:]
            tag = l[0].split(':')[0]
            if tags.has_key(tag):
                tags[tag] += 1
            else:
                tags[tag] = 0
            #{u'LOC': 834, u'HUM': 1222, u'NUM': 895, u'ABBR': 85, u'ENTY': 1249, u'DESC': 1161}

            if len(words) > max_len:
                max_len = len(words)

            for word in words:
                if self.lower:
                    word = word.lower()
                if not frequences.has_key(word):
                    frequences[word] = 1
                else:
                    frequences[word] = frequences[word] + 1
        # print tags
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

        print ' ', 'Max len', max_len
        print ' ', 'Max vocabulary size (include especial tokens)', len(self.token2id)

        # cPickle.dump(self.token2id.keys(),open('data/trec_vocab.pkl', 'wb'))

        return {'size':len(self.token2id), 'token2id':self.token2id, 'id2token':self.id2token, 'freq':self.frequences}

    def makeData(self, filepath, train=False):
        lines = codecs.open(filepath, 'r', 'utf-8').readlines()
        s_data = []
        l_data = []
        sizes = []
        _i = 0
        label_dic = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        for line in lines:
            _i += 1
            # print _i
            line = line.lower() if self.lower else line
            l = line.strip().split(' ')

            words = l[1:]
            tag = l[0].split(':')[0]

            if tag == 'LOC' or tag == 'loc':
                label = 0
                label_dic[0] += 1
            elif tag == 'HUM' or tag == 'hum':
                label = 1
                label_dic[1] += 1
            elif tag == 'NUM' or tag == 'num':
                label = 2
                label_dic[2] += 1
            elif tag == 'ABBR' or tag == 'abbr':
                label = 3
                label_dic[3] += 1
            elif tag == 'ENTY' or tag == 'enty':
                label = 4
                label_dic[4] += 1
            elif tag == 'DESC' or tag == 'desc':
                label = 5
                label_dic[5] += 1

            s_data.append(torch.LongTensor([self.token2id[word] if self.token2id.has_key(word) else constant.UNK_ID for word in words]))
            l_data.append(torch.LongTensor([label]))
            sizes.append(len(words))
        # print label_dic

        if self.shuffle and train:
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

        return s_data, l_data

    def wordvec(self, flag):
        lines = codecs.open('data/trec_data/raw_data/trec_glove_300d.txt', 'r', 'utf-8').readlines()
        wordvec = {}
        for line in lines:
            l = line.strip().split()
            word = l[0]
            vec = [float(i) for i in l[1:]]
            wordvec[word] = vec
        # print self.id2token[0], wordvec.has_key(self.id2token[0])
        # print self.id2token[1], wordvec.has_key(self.id2token[1])
        # print self.id2token[2], wordvec.has_key(self.id2token[2])
        # print self.id2token[9776], wordvec.has_key(self.id2token[9776])

        wv = np.zeros((len(self.token2id), 300))
        count = 0
        for i in range(len(self.token2id)):
            word = self.id2token[i]
            if wordvec.has_key(word):
                count += 1
            vec = wordvec[word] if wordvec.has_key(word) else np.random.uniform(-0.5, 0.5, size=(1, 300))
            wv[i] = vec
        print ' ', count, 'found in Glove'
        outfile = 'wv_all_' + flag + '.npz'
        print '* Saving data to \'' + self.save_data + outfile + '...'
        # print wv[0][:10]
        # print wv[1][:10]
        # print wv[9776][:10]
        wv.dump(self.save_data + outfile)

    def wordvec2(self, flag):
        lines = codecs.open('data/trec_deps_300d.txt', 'r', 'utf-8').readlines()
        wordvec = {}
        for line in lines:
            l = line.strip().split()
            word = l[0]
            vec = [float(i) for i in l[1:]]
            wordvec[word] = vec
        # print self.id2token[0], wordvec.has_key(self.id2token[0])
        # print self.id2token[1], wordvec.has_key(self.id2token[1])
        # print self.id2token[2], wordvec.has_key(self.id2token[2])
        # print self.id2token[9776], wordvec.has_key(self.id2token[9776])

        wv = np.zeros((len(self.token2id), 300))
        count = 0
        for i in range(len(self.token2id)):
            word = self.id2token[i]
            if wordvec.has_key(word):
                count += 1
            vec = wordvec[word] if wordvec.has_key(word) else np.random.uniform(-0.1, 0.1, size=(1, 300))
            wv[i] = vec
        print ' ', count, 'found in Deps'
        outfile = 'wv_all_deps_' + flag + '.npz'
        print '* Saving data to \'' + self.save_data + outfile + '...'
        # print wv[0][:10]
        # print wv[1][:10]
        # print wv[9776][:10]
        wv.dump(self.save_data + outfile)

    def wordvec3(self, flag):
        lines = codecs.open('data/trec_google_300d.txt', 'r', 'utf-8').readlines()
        wordvec = {}
        for line in lines:
            l = line.strip().split()
            word = l[0]
            vec = [float(i) for i in l[1:]]
            wordvec[word] = vec
        # print self.id2token[0], wordvec.has_key(self.id2token[0])
        # print self.id2token[1], wordvec.has_key(self.id2token[1])
        # print self.id2token[2], wordvec.has_key(self.id2token[2])
        # print self.id2token[9776], wordvec.has_key(self.id2token[9776])

        wv = np.zeros((len(self.token2id), 300))
        count = 0
        for i in range(len(self.token2id)):
            word = self.id2token[i]
            if wordvec.has_key(word):
                count += 1
            vec = wordvec[word] if wordvec.has_key(word) else np.random.uniform(-0.1, 0.1, size=(1, 300))
            wv[i] = vec
        print ' ', count, 'found in google'
        outfile = 'wv_all_google_' + flag + '.npz'
        print '* Saving data to \'' + self.save_data + outfile + '...'
        # print wv[0][:10]
        # print wv[1][:10]
        # print wv[9776][:10]
        wv.dump(self.save_data + outfile)

    def process(self):

        time_stamp = time.strftime("%Y_%m_%d", time.localtime()) 

        print '* Build vocabulary...'
        vocab = self.makeVocab(self.corpus_src)
        print '* Bulid OK.'

        print '* Prepare training data...'
        train = self.makeData(self.train_src, True)

        print '  Prepare valid data...'
        valid = self.makeData(self.valid_src)

        print '  Prepare test data...'
        test = self.makeData(self.test_src)
        print '* Prepare OK.'

        outfile = 'train_6_all_' + time_stamp + '.pt'
        print '* Saving data to \'' + self.save_data + outfile + '...'
        save_data = {'vocab': vocab,
                     'train': train,
                     'valid': valid,
                     'test': test}
        torch.save(save_data, self.save_data + outfile)
        print '* OK.'

        # get pre-trained wordvec
        print '* Find pre-trained wordvec...'
        self.wordvec(time_stamp)
        print '* OK.'

        # print '* Find pre-trained wordvec...'
        # self.wordvec2(time_stamp)
        # print '* OK.'

        # print '* Find pre-trained wordvec...'
        # self.wordvec3(time_stamp)
        # print '* OK.'

def train_split(file, num):
    print '* Splitting training data...'
    lines = codecs.open(file, 'r', 'utf-8').readlines()
    index = range(len(lines))
    np.random.shuffle(index)
    count = 0
    train = []
    valid = []
    for i in index:
        if count < num:
            valid.append(lines[i])
            count += 1
        else:
            train.append(lines[i])

    print '  %d training samples\n  %d new training samples, %d valid samples' \
                        % (len(index), len(train), len(valid))

    train_file = 'data/trec_split_train.txt'
    f = codecs.open(train_file, 'w', 'utf-8')
    f.write(''.join(train))
    f.close()

    valid_file = 'data/trec_split_valid.txt'
    f = codecs.open(valid_file, 'w', 'utf-8')
    f.write(''.join(valid))
    f.close()

    print '* OK.'

# train_split('data/trec_train.txt', 500)

p = PreProcess(shuffle=True)
p.process()




