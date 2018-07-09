#coding=utf-8
import torch
from torch.autograd import Variable
import constant
import math

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, opt, data, dicts, volatile=False):
        super(Dataset, self).__init__()

        self.tokens = data[0]
        self.labels = data[1]

        # print self.tokens[0]
        # print [dicts['id2token'][i] for i in self.tokens[0]]
        # print self.labels[0]

        self.batch_size = opt['batch_size']
        self.num_batches = (len(self.tokens)/self.batch_size + 1) \
                            if len(self.tokens)/self.batch_size < float(len(self.tokens))/self.batch_size \
                            else (len(self.tokens)/self.batch_size)

        self.cuda = opt['gpu']

        self.lstm = opt['lstm']
        
        self.volatile = volatile

        self.shuffled_tokens = self.tokens[:]
        self.shuffled_labels = self.labels[:]

        self.train_indices = None
        self.shuffle()

    def _batch_t(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(constant.PAD_ID)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, tuple(lengths)
        else:
            return out

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        max_index = None
        if (index+1)*self.batch_size < len(self.labels):
            max_index = (index+1)*self.batch_size
        else:
            max_index = len(self.labels)

        batch_tokens, batch_lengths = self._batch_t(
                self.shuffled_tokens[index*self.batch_size:max_index],
                align_right=False, include_lengths=True)
        batch_labels = self.shuffled_labels[index*self.batch_size:max_index]

        if self.lstm:
            indices = range(len(batch_tokens))
            batch = zip(indices, batch_tokens, batch_labels)
            batch, batch_lengths = zip(*sorted(zip(batch, batch_lengths), key=lambda x: -x[1]))
            indices, batch_tokens, batch_labels = zip(*batch)


        # wrap to pytorch Variable
        def wrap(b, vector=False):
            # to sequence length * batch size or sequence length
            if vector:
                b = torch.stack(b, 0).contiguous().squeeze(1)
            else:
                b = torch.stack(b, 0)
            if self.cuda != -1:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b
            
        batch_tokens = wrap(batch_tokens, True)
        batch_labels = wrap(batch_labels)
        
        return ((batch_tokens, batch_lengths), batch_labels)


    def __len__(self):
        return self.num_batches


    def shuffle(self):
        # tracing the disordering sentence, then we can log them with a fixed order
        data = list(zip(self.tokens, self.labels))
        indices = list(torch.randperm(len(data)))
        train_indices = {}
        for i, index in enumerate(indices):
            train_indices[index] = i
        self.shuffled_tokens, self.shuffled_labels \
                = zip(*[data[i] for i in indices])

        self.train_indices = train_indices
        return train_indices
