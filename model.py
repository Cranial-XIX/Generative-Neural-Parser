import copy
import gr
import itertools
import math
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

class LCNPModel(nn.Module):
    """
    The Left Context Neural Parser (LCNP) Model
    """

    def __init__(self, args):
        super(LCNPModel, self).__init__()

        self.verbose = args['verbose']
        self.use_cuda = args['cuda']

        # terminals
        self.nt = args['nt']               # number of terminals
        self.dt = args['dt']               # dimension of terminals

        # nonterminals
        self.nnt = args['nnt']             # number of nonterminals
        self.dnt = args['dnt']             # dimension of nonterminals
        self.dnt = 50
        # model
        self.lstm_coef = args['lstm_coef'] # coefficient of LSTM
        self.bsz = args['bsz']             # the batch size
        self.dhid = args['dhid']           # LSTM hidden dimension size

        self.nunary = args['nunary']
        self.nlayers = args['nlayers']     # number of layers in neural net
        self.lexicon = args['lexicon']     # dictionary for lexicon
        self.parser = args['parser']       # the parser, written in Cython

        # the precomputed matrix that will be used in unsupervised learning
        self.h0 = (
            Variable(torch.zeros(self.nlayers, self.bsz, self.dhid)),
            Variable(torch.zeros(self.nlayers, self.bsz, self.dhid))
        )

        if self.use_cuda:
            # the initial states for h0 and c0 of LSTM
            self.h0 = (
                Variable(torch.zeros(self.nlayers, self.bsz, self.dhid).cuda()),
                Variable(torch.zeros(self.nlayers, self.bsz, self.dhid).cuda())
            )
            # initialize precomputed matrix

        self.h0[0].requires_grad = False
        self.h0[1].requires_grad = False

        # nonterminal embedding and w2v embedding, w2v_plus 
        # is the deviation from w2v
        self.nt_emb = nn.Embedding(self.nnt, self.dnt)
        self.word_emb = nn.Embedding(self.nt, self.dt)

        # The LSTM and some linear transformation layers
        self.LSTM = nn.LSTM(
            self.dt, self.dhid, self.nlayers,
            batch_first=True, bias=True, dropout=0.5
        )

        dp2l = dunt = dut = self.dnt + self.dhid
        dpl2r = 2 * (self.dnt + self.dhid)

        self.lsm = nn.LogSoftmax()
        self.sm = nn.Softmax()
        self.relu = nn.ReLU()

        zeta = 0.55

        hp2l = int(zeta * dp2l + (1-zeta) * self.nnt)
        hpl2r = int(zeta * dpl2r + (1-zeta) * self.nnt)
        hunt = int(zeta * dunt + (1-zeta) * self.nunary)

        # parent to left
        self.p2l = nn.Linear(dp2l, hp2l)
        self.p2l_out = nn.Linear(hp2l, self.nnt)
        # parent left to right
        self.pl2r = nn.Linear(dpl2r, hpl2r)
        self.pl2r_out = nn.Linear(hpl2r, self.nnt)
        # unary nonterminal
        self.unt = nn.Linear(dunt, hunt)
        self.unt_out = nn.Linear(hunt, self.nunary)
        # unary terminal
        self.ut = nn.Linear(dut, 300)
        self.ut_out = nn.Linear(300, self.nt)
        self.init_weights(args['initrange'], args['word_emb'], args['nt_emb'])

    def init_weights(self, initrange, word_emb, nt_emb):
        self.word_emb.weight.data = word_emb
        #self.nt_emb.weight.data = nt_emb
        self.word_emb.weight.requires_grad = False
        #self.nt_emb.weight.requires_grad = False

        # Below are initial setup for LSTM

        lstm_weight_range = 0.2

        self.LSTM.weight_ih_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)
        '''
        self.LSTM.weight_ih_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_ih_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)  
        self.LSTM.weight_hh_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)
        '''

        size = len(self.LSTM.bias_ih_l0)
        section = size / 4
        for i in xrange(section, 2*section):
            self.LSTM.bias_ih_l0.data[i] = 1.0
            self.LSTM.bias_hh_l0.data[i] = 1.0
        '''
            self.LSTM.bias_ih_l1.data[i] = 1.0
            self.LSTM.bias_hh_l1.data[i] = 1.0
            self.LSTM.bias_ih_l2.data[i] = 1.0
            self.LSTM.bias_hh_l2.data[i] = 1.0

        '''

        self.p2l.bias.data.fill_(0)
        self.p2l_out.bias.data.fill_(0)
        #self.p2l.weight.data.uniform_(-initrange, initrange)

        self.pl2r.bias.data.fill_(0)
        #self.pl2r.weight.data.uniform_(-initrange, initrange)

        self.pl2r_out.bias.data.fill_(0)
        #self.pl2r_out.weight.data.uniform_(-initrange, initrange)

        self.unt.bias.data.fill_(0)
        self.unt_out.bias.data.fill_(0)
        #self.unt.weight.data.uniform_(-initrange, initrange)

        self.ut.bias.data.fill_(0)
        #self.ut.weight.data.uniform_(-initrange, initrange)

    def forward(self, train_type, args):
        if train_type == 'supervised':
            return self.supervised(*args)
        elif train_type == 'unsupervised':
            return self.unsupervised(*args)
        else:
            print "Unrecognized train type!"
            return

    def parsing_setup(self):
        # since for lexicon, Pr(x | P) = logsoftmax(A(Wx + b)). We
        # precompute AW (as ut_w) and Ab (as ut_b) here to speed up the computation
        #w2v_w = self.word_emb.weight
        #self.ut_w = w2v_w.mm(self.ut.weight).t()
        #self.ut_b = w2v_w.mm(self.ut.bias.view(-1, 1)).t()
        self.precomputed = self.nt_emb.weight

    def parse(self, sentence, sen, viterbi=True):
        # sen is a torch.LongTensor object, containing fake BOS index
        # along with indices of the words

        t0 = time.time()
        # get left context hidden units with a trained h0 (start hidden state)
        output, hidden = self.LSTM(self.word_emb(sen.view(1,-1)), self.h0)
        # get rid of the initial BOS symbol, since you need fake BOS 
        # to make sure everyone gets their cup of left context
        sen = sen.data[1:]
        n = len(sen)
        # truncate the last left context out, since we need to ensure the
        # matrix is of length n
        # * output (batch_size * sen_length * hidden dimension)
        output = output.narrow(1, 0, n)

        softmax = self.lsm if viterbi else self.sm

        ## pre-compute all probabilities

        # with context probabilities
        # * unt_i (num_nt * sen_length * parent_emb_size)
        cond = torch.cat((
            self.precomputed.unsqueeze(1).repeat(1, n, 1),
            output.repeat(self.nnt, 1, 1) # (num_nt * sen_length * hidden dimension)
        ), 2)

        sz = cond.size()

        # parent to unary child
        # * unt_pr (num_nt * sen_length * num_nt) -> (parent, position i, child)
        unt_pr = softmax(
            self.unt_out(
                self.relu(
                    self.unt(
                        cond.view(-1, sz[2])
                    )
                )
            )
        ).view(sz[0], sz[1], -1)

        p2l_pr = softmax(
            self.p2l_out(
                self.relu(
                    self.p2l(
                        cond.view(-1, sz[2])
                    )
                )
            )
        ).view(sz[0], sz[1], -1)

        output = output.view(n, -1)
        if viterbi:
            preterminal = np.empty((n, self.nnt), dtype=np.float32)
            preterminal.fill(-1000000)
        else:
            preterminal = np.zeros((n, self.nnt), dtype=np.float32)

        U_TM = 0

        t1= time.time()

        self.pos = []
        # append one level preterminal symbols
        for i in xrange(n):
            c = sen[i]
            max = -1000000
            idx = -1
            for p in self.lexicon[c]:
                pi = Variable(torch.LongTensor([p]))
                h = output[i].view(1, -1)
                if self.use_cuda:
                    pi = pi.cuda()
                    h = h.cuda()
                cond = torch.cat((self.nt_emb(pi), h), 1)
                #res = cond.mm(self.ut_w) + self.ut_b
                res = self.ut_out(self.relu(self.ut(cond)))

                x = (softmax(self.p2l(cond))[0][U_TM] + softmax(res)[0][c]).data[0]
                if x > max:
                    idx = p
                    max = x
                preterminal[i,p] = x
            self.pos.append(idx)
        

        t2 = time.time()
        # preprocess
        pl2r_p, pl2r_l, pl2r_pi, pl2r_ci = self.parser.preprocess(n, preterminal)

        pl2r_p = Variable(torch.LongTensor(pl2r_p))
        pl2r_l = Variable(torch.LongTensor(pl2r_l))
        pl2r_pi = Variable(torch.LongTensor(pl2r_pi))
        pl2r_ci = Variable(torch.LongTensor(pl2r_ci))

        if self.use_cuda:
            pl2r_p = pl2r_p.cuda()
            pl2r_l = pl2r_l.cuda()
            pl2r_pi = pl2r_pi.cuda()
            pl2r_ci = pl2r_ci.cuda()

        t3 = time.time()

        parent_lc = torch.index_select(output, 0, pl2r_pi)
        sibling_lc = torch.index_select(output, 0, pl2r_ci)
        pl2r_cond = torch.cat((
            self.nt_emb(pl2r_p),
            self.nt_emb(pl2r_l),
            parent_lc,
            sibling_lc - parent_lc
        ), 1)

        # compute the log probability of pl2r rules
        pl2r_pr = softmax(
            self.pl2r_out(
                self.relu(
                    self.pl2r(
                        pl2r_cond
                    )
                )
            )
        )

        t4 = time.time()
        #print " - "*10, t4-t3, t3-t2, t2-t1, t1-t0
        if self.use_cuda:
            if viterbi:
                return self.parser.viterbi(
                        sentence,
                        preterminal,
                        unt_pr.cpu().data.numpy(),
                        p2l_pr.cpu().data.numpy(),
                        pl2r_pr.cpu().data.numpy()
                )
            else:
                return self.parser.mbr(
                        sentence,
                        preterminal,
                        unt_pr.cpu().data.numpy(),
                        p2l_pr.cpu().data.numpy(),
                        pl2r_pr.cpu().data.numpy()
                )
        else:
            if viterbi:
                return self.parser.viterbi(
                        sentence,
                        preterminal,
                        unt_pr.data.numpy(),
                        p2l_pr.data.numpy(),
                        pl2r_pr.data.numpy()
                )
            else:
                return self.parser.mbr(
                    sentence,
                    preterminal,
                    unt_pr.data.numpy(),
                    p2l_pr.data.numpy(),
                    pl2r_pr.data.numpy()
                )               

    def unsupervised(self, sen):
        # sen is a torch.LongTensor object, containing fake BOS index
        # along with indices of the words

        t0 = time.time()
        # get left context hidden units with a trained h0 (start hidden state)
        output, hidden = self.LSTM(self.word_emb(sen.view(1,-1)), self.h0)
        # get rid of the initial BOS symbol, since you need fake BOS 
        # to make sure everyone gets their cup of left context
        sen = sen.data[1:]
        n = len(sen)
        # truncate the last left context out, since we need to ensure the
        # matrix is of length n
        # * output (batch_size * sen_length * hidden dimension)
        output = output.narrow(1, 0, n)

        softmax = self.lsm if viterbi else self.sm

        # * h1 (num_nt * sen_length * hidden dimension)
        h1 = output.repeat(self.nnt, 1, 1)

        ## pre-compute all probabilities

        # with context probabilities
        # * unt_i (num_nt * sen_length * parent_emb_size) 
        unt_i = self.unt_pre.unsqueeze(1).repeat(1, n, 1)
        p2l_i = self.p2l_pre.unsqueeze(1).repeat(1, n, 1)
        unt_cond = torch.cat((unt_i, h1), 2)
        p2l_cond = torch.cat((p2l_i, h1), 2)
        sz = unt_cond.size()

        # parent to unary child
        # * unt_pr (num_nt * sen_length * num_nt) -> (parent, position i, child)
        unt_pr = softmax(self.unt_out(self.relu(self.unt(unt_cond.view(-1, sz[2]))))).view(sz[0], sz[1], -1)
        p2l_pr = softmax(self.p2l_out(self.relu(self.p2l(p2l_cond.view(-1, sz[2]))))).view(sz[0], sz[1], -1)

        output = output.view(n, -1)
        if viterbi:
            preterminal = np.empty((n, self.nnt), dtype=np.float32)
            preterminal.fill(-1000000)
        else:
            preterminal = np.zeros((n, self.nnt), dtype=np.float32)

        U_TM = 0
        t1= time.time()
        # append one level preterminal symbols
        for i in xrange(n):
            c = sen[i]
            for p in self.lexicon[c]:
                pi = Variable(torch.LongTensor([p]))
                h = output[i].view(1, -1)
                if self.use_cuda:
                    pi = pi.cuda()
                    h = h.cuda()
                cond = torch.cat((self.nt_emb(pi), h), 1)
                res = cond.mm(self.ut_w) + self.ut_b

                preterminal[i,p] = (softmax(self.p2l(cond))[0][U_TM] + softmax(res)[0][c]).data[0]
        

        t2 = time.time()
        # preprocess
        pl2r_p, pl2r_l, pl2r_pi, pl2r_ci = self.parser.preprocess(n, preterminal)

        pl2r_p = Variable(torch.LongTensor(pl2r_p))
        pl2r_l = Variable(torch.LongTensor(pl2r_l))
        pl2r_pi = Variable(torch.LongTensor(pl2r_pi))
        pl2r_ci = Variable(torch.LongTensor(pl2r_ci))

        if self.use_cuda:
            pl2r_p = pl2r_p.cuda()
            pl2r_l = pl2r_l.cuda()
            pl2r_pi = pl2r_pi.cuda()
            pl2r_ci = pl2r_ci.cuda()

        t3 = time.time()

        parent_lc = torch.index_select(output, 0, pl2r_pi)
        sibling_lc = torch.index_select(output, 0, pl2r_ci)
        pl2r_cond = torch.cat((
            self.relu(
                self.pl2r_base(
                    torch.cat((
                        self.nt_emb(pl2r_p),
                        self.nt_emb(pl2r_l),
                        parent_lc,
                    ), 1)
                )
            ),
            sibling_lc - parent_lc
        ), 1)

        # compute the log probability of pl2r rules
        pl2r_pr = softmax(
            self.pl2r_out(
                self.relu(
                    self.pl2r(
                        pl2r_cond
                    )
                )
            )
        )

        t4 = time.time()
        #print " - "*10, t4-t3, t3-t2, t2-t1, t1-t0

        

    def supervised(self, sens,
        p2l, pl2r_p, pl2r_l, unt, ut,
        p2l_t, pl2r_t, unt_t, ut_t,
        p2l_i, pl2r_pi, pl2r_ci, unt_i, ut_i):

        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.word_emb(sens), self.h0)
        output = self.lstm_coef * output.contiguous().view(-1, output.size(2))

        # compute the log probability of p2l rules

        p2l_cond = torch.cat((
            self.nt_emb(p2l), 
            torch.index_select(output, 0, p2l_i)
        ), 1)

        nll_p2l = -torch.sum(
            self.lsm(self.p2l_out(self.relu(self.p2l(p2l_cond)))).gather(1, p2l_t.unsqueeze(1))
        )

        # compute the log probability of unary nonterminal rules
        unt_cond = torch.cat((
            self.nt_emb(unt), 
            torch.index_select(output, 0, unt_i)
        ), 1)

        nll_unt = -torch.sum(
            self.lsm(self.unt_out(self.relu(self.unt(unt_cond)))).gather(1, unt_t.unsqueeze(1))
        )

        # compute the log probability of terminal rules
        ut_cond = torch.cat((
            self.nt_emb(ut),
            torch.index_select(output, 0, ut_i)
        ), 1)

        #m_ut = self.ut(ut_cond).mm(self.word_emb.weight.t())
        nll_ut = -torch.sum(
            self.lsm(
                self.ut_out(self.relu(self.ut(ut_cond)))
            ).gather(1, ut_t.unsqueeze(1))
        )

        # compute the log probability of pl2r rules
        parent_lc = torch.index_select(output, 0, pl2r_pi)
        sibling_lc = torch.index_select(output, 0, pl2r_ci)
        pl2r_cond = torch.cat((
            self.nt_emb(pl2r_p),
            self.nt_emb(pl2r_l),
            parent_lc,
            sibling_lc - parent_lc
        ), 1)

        # pass to a single layer neural net for nonlinearity
        nll_pl2r = -torch.sum(
            self.lsm(
                self.pl2r_out(
                    self.relu(
                        self.pl2r(pl2r_cond)
                    )
                )
            ).gather(1, pl2r_t.unsqueeze(1))
        )

        return nll_p2l + nll_pl2r + nll_unt + nll_ut
        #return nll_ut

    def pl2r_test(self, sens, pl2r_p, pl2r_l, pl2r_t, pl2r_pi, pl2r_ci):
        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.word_emb(sens), self.h0)
        output = self.lstm_coef * output.contiguous().view(-1, output.size(2))

        # compute the log probability of pl2r rules
        pl2r_cond = torch.cat((
            self.nt_emb(pl2r_p),
            self.nt_emb(pl2r_l),
            torch.index_select(output, 0, pl2r_pi),
            torch.index_select(output, 0, pl2r_ci)
        ), 1)

        # pass to a single layer neural net for nonlinearity
        m_pl2r = self.relu(self.pl2r(pl2r_cond))
        return self.sm(self.pl2r_out(m_pl2r))#.gather(1, pl2r_t.unsqueeze(1))

    def p2l_test(self, sens, p2l, p2l_t, p2l_i):

        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.word_emb(sens), self.h0)
        output = self.lstm_coef * output.contiguous().view(-1, output.size(2))

        # compute the log probability of p2l rules
        p2l_cond = torch.cat((
            self.nt_emb(p2l), 
            torch.index_select(output, 0, p2l_i)
        ), 1)

        return self.sm(self.p2l_out(self.relu(self.p2l(p2l_cond))))

    def unt_test(self, sens, unt, unt_t, unt_i):

        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.word_emb(sens), self.h0)
        output = self.lstm_coef * output.contiguous().view(-1, output.size(2))

        # compute the log probability of unary nonterminal rules
        unt_cond = torch.cat((
            self.nt_emb(unt), 
            torch.index_select(output, 0, unt_i)
        ), 1)

        return self.sm(self.unt_out(self.relu(self.unt(unt_cond))))

    def ut_test(self, sens, ut, ut_t, ut_i):

        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.word_emb(sens), self.h0)
        output = self.lstm_coef * output.contiguous().view(-1, output.size(2))

        # compute the log probability of terminal rules
        ut_cond = torch.cat((
            self.nt_emb(ut),
            torch.index_select(output, 0, ut_i)
        ), 1)

        m_ut = self.ut(ut_cond).mm(self.word_emb.weight.t())

        return self.sm(m_ut)
