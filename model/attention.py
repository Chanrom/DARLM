#coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, input_size, dim, gpu):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gpu = gpu

        self.W_m = nn.Linear(dim, input_size, bias=False) # weight: input_size x dim
        self.W_h = nn.Linear(dim, hidden_size, bias=False)
        self.W_c = nn.Linear(dim, input_size, bias=False)
        self.w = nn.Linear(1, dim, bias=False)
        self.dim = dim
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

    def getMask(self, batch_size, lens):
        mask = torch.ByteTensor(batch_size, max(lens)).fill_(1)
        for i in range(batch_size):
            for j in range(lens[i]):
                mask[i][j] = 0 # mask 中为1的

        if self.gpu != -1:
            return mask.cuda()
        else:
            return mask

    def forward(self, htm1, ctm1, memory, lens):
        """
        htm1: batch x hidden, controler lstm previous time output h
        ctm1: batch x input, controler lstm previous time input x
        memory: batch x seqlen x input
        lens: tuple
        """
        batch_size = memory.size(0)
        seq_len = memory.size(1)

        e_L = Variable(htm1.data.new(batch_size, seq_len, 1).fill_(1)) # batch x seqlen x 1

        htm1 = htm1.unsqueeze(1) # batch x 1 x hidden
        Htm1 = torch.bmm(e_L, htm1) # batch x seqlen x hidden
        W_h = self.W_h.weight.unsqueeze(0).expand(batch_size, self.hidden_size, self.dim)
        h_info = torch.bmm(Htm1, W_h) # batch x seqlen x dim

        ctm1 = ctm1.unsqueeze(1) # batch x 1 x input
        Ctm1 = torch.bmm(e_L, ctm1) # batch x seqlen x input
        W_c = self.W_c.weight.unsqueeze(0).expand(batch_size, self.input_size, self.dim)
        c_info = torch.bmm(Ctm1, W_c) # batch x seqlen x dim

        W_m = self.W_m.weight.unsqueeze(0).expand(batch_size, self.input_size, self.dim)
        m_info = torch.bmm(memory, W_m) # batch x seqlen x dim

        Y = self.tanh(h_info + c_info + m_info) # batch x seqlen x dim
        # Y = self.tanh(c_info + m_info) # batch x seqlen x dim

        w = self.w.weight.unsqueeze(0).expand(batch_size, self.dim, 1) # batch x dim x 1

        Alpha = torch.bmm(Y, w).squeeze(2) # batch x seqlen

        mask = self.getMask(batch_size, lens)
        Alpha.data.masked_fill_(mask, -float('inf'))

        Alpha = self.sm(Alpha)
        Alpha3 = Alpha.view(batch_size, 1, seq_len)  # batch x 1 x seqlen

        weightedContext = torch.bmm(Alpha3, memory).squeeze(1)  # batch x dim

        return weightedContext, Alpha


class RecursiveAttention(nn.Module):
    """refer to Long Short-Term Memory-Networks for Machine Reading"""
    def __init__(self, mem_size, input_size, gpu):
        super(RecursiveAttention, self).__init__()
        self.mem_size = mem_size
        self.input_size = input_size
        self.gpu = gpu

        self.W_m = nn.Linear(mem_size, input_size, bias=False) # weight: input_size x dim
        self.W_h = nn.Linear(mem_size, mem_size, bias=False)
        self.W_h_hat = nn.Linear(mem_size, mem_size, bias=False)
        self.w = nn.Linear(1, mem_size, bias=False)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

    def getMask(self, batch_size, lens, time):
        mask = torch.ByteTensor(batch_size, time).fill_(1)
        for i in range(batch_size):
            max_len = time if time < (lens[i] - 1) else lens[i] - 1
            for j in range(max_len):
                mask[i][j] = 0 # mask 中为1的

        if self.gpu != -1:
            return mask.cuda()
        else:
            return mask

    def forward(self, all_htm1, all_ctm1, h_hattm1, xt, lens, time):
        """
        all_htm1: batch x max(until_lens) x hidden, memory output for each time stamp
        h_hattm1: batch x mem_size, memory context output for previous time stamp
        xt: batch x input_size
        lens: input lens
        time: int, current time stamp index
        """
        batch_size = all_htm1.size(0)
        seq_len = all_htm1.size(1)
        # print 'attn seqlen', seq_len, batch_size
        # print all_htm1.size(), all_ctm1.size(), h_hattm1.size(), xt.size()

        e_L = Variable(h_hattm1.data.new(batch_size, seq_len, 1).fill_(1)) # batch x seqlen x 1
        # print 'attn e_L', e_L.size()

        h_hattm1 = h_hattm1.unsqueeze(1) # batch x 1 x hidden
        H_hattm1 = torch.bmm(e_L, h_hattm1) # batch x seqlen x hidden
        W_h_hat = self.W_h_hat.weight.unsqueeze(0).expand(batch_size, self.mem_size, self.mem_size)
        h_hatinfo = torch.bmm(H_hattm1, W_h_hat) # batch x seqlen x hidden

        xt = xt.unsqueeze(1) # batch x 1 x input
        Xt = torch.bmm(e_L, xt) # batch x seqlen x input
        W_m = self.W_m.weight.unsqueeze(0).expand(batch_size, self.input_size, self.mem_size)
        x_info = torch.bmm(Xt, W_m) # batch x seqlen x dim

        W_h = self.W_h.weight.unsqueeze(0).expand(batch_size, self.mem_size, self.mem_size)
        h_info = torch.bmm(all_htm1, W_h) # batch x seqlen x dim

        Y = self.tanh(h_info + x_info + h_hatinfo) # batch x seqlen x dim
        # Y = self.tanh(c_info + m_info) # batch x seqlen x dim

        w = self.w.weight.unsqueeze(0).expand(batch_size, self.mem_size, 1) # batch x dim x 1

        Alpha = torch.bmm(Y, w).squeeze(2) # batch x seqlen

        mask = self.getMask(batch_size, lens, time)
        # print 'mask', mask.size(), time
        Alpha.data.masked_fill_(mask, -float('inf'))

        Alpha = self.sm(Alpha)
        Alpha3 = Alpha.view(batch_size, 1, seq_len)  # batch x 1 x seqlen

        h_hatt = torch.bmm(Alpha3, all_htm1).squeeze(1)  # batch x dim
        c_hatt = torch.bmm(Alpha3, all_ctm1).squeeze(1)  # batch x dim

        return h_hatt, c_hatt, Alpha