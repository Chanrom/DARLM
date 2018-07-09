#coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
from torch.autograd import Variable
# from torch.nn.utils.rnn import pad_packed_sequence as unpack
# from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils import constant
from attention import *

class Memory(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Memory, self).__init__()

        self.pre_word_vecs = opt['pre_word_vecs']
        self.word_vec_size = opt['word_vec_size']

        self.word_lut = nn.Embedding(vocab_size,
                                  opt['word_vec_size'],
                                  padding_idx=constant.PAD_ID)

        self.lstm = opt['lstm']

        self.mem_size = opt['mem_size']

        self.use_batch_norm = opt['use_batch_norm']

        if not self.lstm:
            self.kernel_sizes = [int(i) for i in opt['kernel_size'].split(',')]

            self.conv1 = nn.Conv2d(1, self.mem_size/3,
                             (self.kernel_sizes[0], self.word_vec_size),
                             padding=(self.kernel_sizes[0]/2, 0))
            self.conv2 = nn.Conv2d(1, self.mem_size/3,
                             (self.kernel_sizes[1], self.word_vec_size),
                             padding=(self.kernel_sizes[1]/2, 0))
            self.conv3 = nn.Conv2d(1, self.mem_size/3,
                             (self.kernel_sizes[2], self.word_vec_size),
                             padding=(self.kernel_sizes[2]/2, 0))

            if self.use_batch_norm:
                self.batch_norm1 = nn.BatchNorm2d(self.mem_size/3)
                self.batch_norm2 = nn.BatchNorm2d(self.mem_size/3)
                self.batch_norm3 = nn.BatchNorm2d(self.mem_size/3)

            self.relu = nn.ReLU()
        else:
            self.layers = opt['layer']
            self.num_directions = 2 if opt['brnn'] else 1
            assert self.mem_size % self.num_directions == 0
            self.hidden_size = self.mem_size // self.num_directions

            self.rnn = nn.LSTM(self.word_vec_size, self.hidden_size,
                            num_layers=self.layers,
                            dropout=opt['dropout'],
                            bidirectional=opt['brnn'])

        self.dropout = nn.Dropout(opt['dropout'])

        self.gpu = opt['gpu']


    def load_pre_word_vecs(self):
        if self.pre_word_vecs is not None:
            print '* Load pre-trained wordvec...'
            pretrained = np.load(self.pre_word_vecs)
            self.word_lut.weight.data.copy_(torch.from_numpy(pretrained))
            print '* Initial wordvec OK.'


    def makeMask(self, batch_size, lens):
        mask = torch.FloatTensor(batch_size, max(lens)).fill_(0)
        for i in range(batch_size):
            for j in range(lens[i]):
                mask[i][j] = 1
        if self.gpu != -1:
            return Variable(mask).cuda()
        else:
            return Variable(mask)


    def forward(self, input):

        lens = input[1]
        # print 'len', lens
        batch_size = input[0].size(0)

        out = None
        if not self.lstm:
            ## word embedding
            emb_out = self.word_lut(input[0])
            # print 'emb', emb_out.size()
            emb_out = self.dropout(emb_out)  # batch * seq * wordvec

            cnn_out1 = self.conv1(emb_out.unsqueeze(1))
            # 去掉最后一列
            cnn_out2 = self.conv2(emb_out.unsqueeze(1))[:, :, :-1,]
            cnn_out3 = self.conv3(emb_out.unsqueeze(1))

            if self.use_batch_norm:
                cnn_out1 = self.batch_norm1(cnn_out1.contiguous())
                cnn_out2 = self.batch_norm2(cnn_out2.contiguous())
                cnn_out3 = self.batch_norm3(cnn_out3.contiguous())

            cnn_out = torch.cat([cnn_out1, cnn_out2, cnn_out3], 1)

            cnn_out = self.relu(cnn_out)
            hidden_size = cnn_out.size(1)
            mask = self.makeMask(batch_size, lens)
            mask = torch.transpose(mask.unsqueeze(2).expand(batch_size, max(lens), hidden_size), 1, 2).unsqueeze(3)
            cnn_out = cnn_out * mask

            out = torch.transpose(cnn_out.squeeze(3), 1, 2) # batch_size * seq_len * hidden_size

        else:
            # transporse because of LSTM accepting seq_len*batch_size
            emb_out = self.word_lut(input[0].t())
            emb_out = self.dropout(emb_out)
            emb_out = pack(emb_out, input[1])

            outputs, hidden_t = self.rnn(emb_out)
            if isinstance(input, tuple):
                outputs, lens = unpack(outputs)

            out = outputs.transpose(0, 1) # batch_size * seq_len * hidden_size

        memory = self.dropout(out)

        return memory # batch_size * seq_len * hidden_size


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1) # return shape: layer x batch * hidden


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        
        self.classes = opt['classes']
        self.input_size = opt['mem_size']
        self.hidden_size = opt['rnn_size_c']
        self.layers = opt['layer_c']

        self.attn = GlobalAttention(self.hidden_size,
                                    self.input_size,
                                    self.hidden_size, # intermediate dim
                                    opt['gpu'])

        self.rnn = StackedLSTMCell(self.layers, self.input_size,
                               self.hidden_size, opt['dropout'])

        self.dropout = nn.Dropout(opt['dropout'])

        self.mlp = nn.Linear(self.hidden_size, self.classes)

        self.sm = nn.Softmax()


    def forward(self, memory, lengths, h_tm1, context_tm1, ch_tm1):
        '''
        memory: batch x seqlen x input_size
        lengths: tuple
        h_tm1: batch x input_size
        context_tm1: batch_size x input_size
        ch_tm1: tuple, (layer * batch * hidden, layer * batch * hidden)
        '''
        context_t, attn = self.attn(h_tm1,
                                    context_tm1,
                                    memory,
                                    lengths)

        output, ch_t = self.rnn(context_t, ch_tm1)

        output_dp = self.dropout(output)

        output_mlp = self.mlp(output_dp)

        output = self.dropout(output)

        output_sm = self.sm(output_mlp)

        return output_sm, attn, ch_t, context_t


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.input_size = 2*opt['rnn_size_c']
        self.output_size = opt['hop']
        self.choose1 = nn.Linear(self.input_size, opt['rnn_size_c'])
        self.choose2 = nn.Linear(opt['rnn_size_c'], self.output_size)
        self.dropout = nn.Dropout(0)
        self.sigmoid = nn.Tanh()
        self.sm = nn.Softmax()

    def forward(self, input):
        '''
        input: list of multiply classifier's LSTM output hs, [batch x hidden, batch x hidden, ...]
               for now we have fixed number of two classifier
        '''
        h0 = input[0]
        h1 = input[1]
        # h_init = input[2]
        # print h0[0][:10]
        # print h1[0][:10]
        h_sub = h0 - h1
        h_times = h0 * h1
        batch_size = h0.size(0)
        hidden_size = h0.size(1)
        feature = torch.cat([h0, h1], 1) # batch x 2*hidden
        feature_dp = self.dropout(feature)
        feature_dp = self.choose1(feature_dp)
        feature_dp = self.sigmoid(feature_dp)
        feature_dp = self.dropout(feature_dp)
        pred = self.sm(self.choose2(feature_dp))

        return pred


class Connector(nn.Module):
    def __init__(self, opt):
        super(Connector, self).__init__()

        self.hidden_size = opt['rnn_size_c']
        self.gpu = opt['gpu']
        self.num_layer = opt['layer_c']

        self.init_hf = nn.Linear(opt['mem_size'], self.num_layer*self.hidden_size)

    def forward(self, memory, lens):
        # memory: batch * sequence * hidden
        # lens: tuple of sentence len
        lens = Variable(torch.FloatTensor(list(lens)))
        if self.gpu != -1:
            lens = lens.cuda()

        avg_m = memory.sum(1).squeeze(1) # batch x hidden
        lens = lens.unsqueeze(1).expand_as(avg_m)
        avg_m = torch.div(avg_m, lens) # batch x hidden

        h = self.init_hf(avg_m) # batch x (layers*hidden)
        # c = self.init_cf(avg_m)

        batch_size = avg_m.size(0)
        h = h.view(batch_size, self.num_layer, self.hidden_size).transpose(0, 1) # layers x batch x hidden
        # c = c.view(batch_size, self.num_layer, self.hidden_size).t()

        # return h, c
        return h # first attention trigger
