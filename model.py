#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.nn.init as nninit
import time

# 29.08.2017
# When processing a very long sequence in number of steps,
# We use the value of the previous hidden state to initialize but detach from the history to make it trainable
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# 28.08.2017
# "char-ngram", "morpheme" or "oracle" (morphological analysis)
class AdditiveModel(nn.Module):
    def __init__(self, args, is_testing=False):
        super(AdditiveModel, self).__init__()
        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.model = model = args.model
        self.subword_vocab_size = subword_vocab_size = args.subword_vocab_size
        self.unit = args.unit
        self.dtype = args.dtype
        self.otype = args.otype
        self.use_cuda = args.use_cuda
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size
        out_vocab_size = args.out_vocab_size

        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        # Language model is one direction lstm
        # Input is a binary vector of size subword_vocab_size
        # We don't learn an embedding for each subword...
        # Change it in the future ??
        # apply the dropout operator only to the non-recurrent connections: Zaremba
        self.lm_lstm = nn.LSTM(subword_vocab_size, \
                               self.rnn_size, \
                               num_layers=args.num_layers,\
                               bidirectional=False, \
                               batch_first=True,\
                               dropout=(1-args.keep_prob)
                            )
        self.dropout = nn.Dropout(1-args.keep_prob)
        # The linear layer that maps from hidden state space to output vocabulary space
        # out_vocab_size is default 5K
        # The most frequent 5K words in the training corpus (it is also a way to regularize parameters)
        self.hidden2word = nn.Linear(self.rnn_size, out_vocab_size)
        # Initialize hidden state with zeros
        self.lm_hidden = self.init_hidden(args.num_layers,numdirec=1)
        # Initialize weights uniformly
        self.init_weights(args.param_init_type,args.init_scale)
        self.init_forget_gates(value=0.)

    def init_hidden(self,numlayer,numdirec=1):
        result = (autograd.Variable(torch.zeros(numlayer*numdirec, self.batch_size, self.rnn_size).type(self.dtype)),
                  autograd.Variable(torch.zeros(numlayer*numdirec, self.batch_size, self.rnn_size).type(self.dtype)))
        return result


    def init_weights(self,init_type,init_scale):
        # Initialize weight matrix
        for p in self.parameters():
            if init_type=="orthogonal" and p.dim()>=2:
                nninit.orthogonal(p)
            elif init_type=="uniform":
                p.data.uniform_(-init_scale, init_scale)
            elif init_type=="xavier_n" and p.dim()>=2:
                nninit.xavier_normal(p)
            elif init_type=="xavier_u" and p.dim()>=2:
                nninit.xavier_uniform(p)
        # Initialize bias for the linear layer
        self.hidden2word.bias.data.fill_(0.0)

    # In 2014 Zarembi paper it is initialized to 0, but
    # TF tutorial says 1 may give better results ?
    def init_forget_gates(self, value=1.):
        for names in self.lm_lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lm_lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(value)

    def forward(self, batch):
        # input dropout - not sure about that
        word_embeds = self.dropout(batch)
        #word_embeds = batch
        lstm_out, self.lm_hidden = self.lm_lstm(word_embeds, self.lm_hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = self.dropout(lstm_out)
        wordvec_space = self.hidden2word(lstm_out.view(-1,lstm_out.size(2)))
        word_scores = F.log_softmax(wordvec_space)
        return word_scores


class WordModel(nn.Module):
    def __init__(self, args):
        super(WordModel, self).__init__()
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
    def forward(self, batch):
        return

# 01.09.2017 - 03.09.2017
# BiLSTM composition
class BiLSTMModel(nn.Module):
    def __init__(self, args, is_testing=False):
        super(BiLSTMModel, self).__init__()
        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.bilstm_num_steps = bilstm_num_steps = args.bilstm_num_steps
        self.unit = args.unit
        self.dtype = args.dtype
        self.otype = args.otype
        self.use_cuda = args.use_cuda
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size
        self.word_dim = args.word_dim
        subword_vocab_size = args.subword_vocab_size
        out_vocab_size = args.out_vocab_size

        if args.unit == 'char':
            subword_dim = args.char_dim
        elif args.unit == 'char-ngram' or args.unit == 'morpheme' or args.unit == 'oracle':
            subword_dim = args.morph_dim
        self.subword_dim = subword_dim
        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        ####### C2W
        # subword embedding lookup
        self.subEmbeddings = nn.Embedding(subword_vocab_size, subword_dim, padding_idx=0)
        self.comp_lstm = nn.LSTM(subword_dim, \
                               self.rnn_size, \
                               num_layers=1,\
                               bidirectional=True, \
                               batch_first=True,\
                               dropout=(1-args.keep_prob)
                            )
        # word embedding is calculated as
        # w_t = W_f.h_forward + W_b.h_backward + b
        self.W_f = nn.Parameter(torch.randn(self.rnn_size, self.word_dim))
        self.W_b = nn.Parameter(torch.randn(self.rnn_size, self.word_dim))
        self.we_bias = nn.Parameter(torch.randn(self.word_dim))

        ######## LSTM LM
        # Language model is one direction lstm
        # Input is a binary vector of size subword_vocab_size
        self.lm_lstm = nn.LSTM(self.word_dim, \
                               self.rnn_size, \
                               num_layers=args.num_layers,\
                               bidirectional=False, \
                               batch_first=True,\
                               dropout=(1-args.keep_prob)
                            )
        self.dropout = nn.Dropout(1-args.keep_prob)
        self.hidden2word = nn.Linear(self.rnn_size, out_vocab_size)
        # Initialize composition hidden states with zeros
        self.comp_hidden = self.init_hidden(numlayer=1, numdirec=2, batchsize=(self.batch_size*self.num_steps))
        # Initialize LM hidden state with zeros
        self.lm_hidden = self.init_hidden(args.num_layers,numdirec=1, batchsize=self.batch_size)
        # Initialize weights uniformly
        self.init_weights(args.param_init_type,args.init_scale)
        self.init_forget_gates(value=0.)

    def init_hidden(self, numlayer, numdirec=1, batchsize=32):
        result = (autograd.Variable(torch.zeros(numlayer*numdirec, batchsize, self.rnn_size).type(self.dtype)),
                  autograd.Variable(torch.zeros(numlayer*numdirec, batchsize, self.rnn_size).type(self.dtype)))
        return result


    def init_weights(self,init_type,init_scale):
        # Initialize weight matrix
        for p in self.parameters():
            if init_type=="orthogonal" and p.dim()>=2:
                nninit.orthogonal(p)
            elif init_type=="uniform":
                p.data.uniform_(-init_scale, init_scale)
            elif init_type=="xavier_n" and p.dim()>=2:
                nninit.xavier_normal(p)
            elif init_type=="xavier_u" and p.dim()>=2:
                nninit.xavier_uniform(p)
        # Initialize bias for the linear layer
        self.hidden2word.bias.data.fill_(0.0)

    # In 2014 Zarembi paper it is initialized to 0, but
    # TF tutorial says 1 may give better results ?
    def init_forget_gates(self, value=1.):
        for names in self.lm_lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lm_lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(value)


    def forward(self, batch):
        #### C2W
        batch = batch.view(self.batch_size*self.num_steps,-1)
        sub_embeds = self.subEmbeddings(batch)
        _, self.comp_hidden = self.comp_lstm(sub_embeds, self.comp_hidden)
        h_n_f = self.comp_hidden[0][0]
        h_n_b = self.comp_hidden[0][1]
        exp_bias = self.we_bias.unsqueeze(0)
        exp_bias = exp_bias.expand(self.batch_size*self.num_steps, self.word_dim)
        word_embeds = torch.mm(h_n_f,self.W_f)+torch.mm(h_n_b,self.W_b)+exp_bias
        word_embeds = word_embeds.view(self.batch_size, self.num_steps, -1)
        #### LM
        word_embeds = self.dropout(word_embeds)
        lstm_out, self.lm_hidden = self.lm_lstm(word_embeds, self.lm_hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = self.dropout(lstm_out)
        wordvec_space = self.hidden2word(lstm_out.view(-1,lstm_out.size(2)))
        word_scores = F.log_softmax(wordvec_space)
        return word_scores
