import copy
import itertools
import math
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

class LCNPModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, inputs, cuda_flag, verbose_flag):
        super(LCNPModel, self).__init__()

        self.verbose = verbose_flag

        # terminals
        self.term_emb = inputs['term_emb']   # embeddings of terminals
        self.nt = inputs['nt']               # number of terminals
        self.dt = inputs['dt']               # dimension of terminals

        # nonterminals
        self.nonterm_emb = inputs['nt_emb']  # embeddings of nonterminals
        self.nnt = inputs['nnt']             # number of nonterminals
        self.dnt = inputs['dnt']             # dimension of nonterminals

        # model
        self.cuda_flag = cuda_flag

        self.coef_lstm = inputs['coef_lstm'] # coefficient of LSTM
        self.bsz = inputs['bsz']             # the batch size
        self.dhid = inputs['dhid']           # dimension of hidden layer
        self.nlayers = inputs['nlayers']     # number of layers in neural net
        initrange = inputs['initrange']      # range for uniform initialization
        self.urules = inputs['urules']       # dictionary of unary rules
        self.brules = inputs['brules']       # dictionary of binary rules
        self.lexicon = inputs['lexicon']     # dictionary of lexicon

        # the precomputed matrix that will be used in unsupervised learning
        self.unt_pre = Variable(inputs['unt_pre'].unsqueeze(1))
        self.p2l_pre = Variable(inputs['p2l_pre'].unsqueeze(1))
        self.pl2r_pre = Variable(inputs['pl2r_pre'].unsqueeze(2))

        self.encoder_nt = nn.Embedding(self.nnt, self.dnt)
        self.word2vec_plus = nn.Embedding(self.nt, self.dt)
        self.word2vec = nn.Embedding(self.nt, self.dt)

        self.LSTM = nn.LSTM(
                self.dt, self.dhid, self.nlayers,
                batch_first=True, dropout=0.5, bias=True
            )

        # the initial states for h0 and c0 of LSTM
        if self.cuda_flag:
            self.h0 = (Variable(torch.zeros(self.nlayers, self.bsz, self.dhid).cuda()),
                Variable(torch.zeros(self.nlayers, self.bsz, self.dhid)).cuda())
        else:
            self.h0 = (Variable(torch.zeros(self.nlayers, self.bsz, self.dhid)),
                Variable(torch.zeros(self.nlayers, self.bsz, self.dhid)))
        

        self.dp2l = self.dnt + self.dhid
        self.dpl2r = 2 * self.dnt + self.dhid
        self.dunt = self.dnt + self.dhid
        self.dut = self.dnt + self.dhid

        # parent to left
        self.p2l = nn.Linear(self.dp2l, self.nnt)
        # parent left to right
        self.pl2r = nn.Linear(self.dpl2r, self.nnt)
        # unary nonterminal
        self.unt = nn.Linear(self.dunt, self.nnt)
        # unary terminal
        self.ut = nn.Linear(self.dut, self.dt)

        self.init_weights(initrange)

    def init_weights(self, initrange=1.0):
        self.word2vec_plus.weight.data.fill_(0)
        self.word2vec.weight.data = self.term_emb
        self.encoder_nt.weight.data = self.nonterm_emb      

        self.word2vec.weight.requires_grad = False
        self.encoder_nt.weight.requires_grad = False 

        # Below are initial setup for LSTM
        lstm_weight_range = 0.2

        self.LSTM.weight_ih_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_ih_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_ih_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)

        size = len(self.LSTM.bias_ih_l1)
        section = size / 4
        for i in xrange(section, 2*section):
            self.LSTM.bias_ih_l0.data[i] = 1.0
            self.LSTM.bias_ih_l1.data[i] = 1.0
            self.LSTM.bias_ih_l2.data[i] = 1.0
            self.LSTM.bias_hh_l0.data[i] = 1.0
            self.LSTM.bias_hh_l1.data[i] = 1.0
            self.LSTM.bias_hh_l2.data[i] = 1.0

        self.p2l.bias.data.fill_(0)
        self.p2l.weight.data.uniform_(-initrange, initrange)

        self.pl2r.bias.data.fill_(0)
        self.pl2r.weight.data.uniform_(-initrange, initrange)

        self.unt.bias.data.fill_(0)
        self.unt.weight.data.uniform_(-initrange, initrange)

        self.ut.bias.data.fill_(0)
        self.ut.weight.data.uniform_(-initrange, initrange)

    def forward(self, sen, *args):
        if len(args) > 1:
            return self.supervised(sen,
                    args[0], args[1], args[2], args[3],
                    args[4], args[5], args[6], args[7],
                    args[8], args[9], args[10], args[11]
                )
        else:
            return self.unsupervised(sen)

    def encoder_t(self, seq):
        return self.word2vec_plus(seq) + self.word2vec(seq)

    def parse(self, sen):

        emb_inp = self.encoder_t(sen)
        output, hidden = self.LSTM(emb_inp, self.h0)
        sen = sen.view(-1).data[1:]
        length = len(sen)
        h = output.squeeze(0).narrow(0, 0, length) # left context

        ## Viterbi Algorithm

        cky = [[[] for j in xrange(length+1)] for i in xrange(length)]
        bp = [[[None for k in xrange(self.nnt)] for j in xrange(length+1)] for i in xrange(length)]
        viscore = [[[0 for k in xrange(self.nnt)] for j in xrange(length+1)] for i in xrange(length)]

        lsm = nn.LogSoftmax()
        w2v_w = self.word2vec.weight + self.word2vec_plus.weight
        ut_w = w2v_w.mm(self.ut.weight).t()
        ut_b = w2v_w.mm(self.ut.bias.view(-1, 1)).t()

        filter = Variable(torch.FloatTensor([-20]))

        # Initialize the chart
        tt0 = time.time()
        for start in xrange(length):
            end = start + 1
            c = sen[start]
            for p in self.lexicon[c]:
                prob = self.log_prob_ut(lsm, ut_w, ut_b, p, c, h[start])
                if prob > filter:
                    cky[start][end].append(p)
                    viscore[start][end][p] = prob

        tt1 = time.time()
        if self.verbose == 'yes':
            print "LEXICON ", tt1-tt0, "---------------------------"

        # Unary appending, deal with non_term -> non_term ... -> term chain
        tt2 = time.time()
        for start in xrange(length):
            end = start + 1
            tmp = []
            for c in cky[start][end]:
                if c in self.urules:
                    for p in self.urules[c]:
                        if p == c:
                            continue
                        newProb = self.log_prob_unt(lsm, p, c, h[start]) + viscore[start][end][c]
                        if newProb > self.max(viscore[start][end][p], filter):
                            if viscore[start][end][p] == 0:
                                tmp.append(p)
                            viscore[start][end][p] = newProb
                            bp[start][end][p] = (None, None, c)
                            print "nonterminal ", p, " has prob ", newProb
            cky[start][end] += tmp

        tt3 = time.time()
        if self.verbose == 'yes':
            print "UNARY ", tt3-tt2, "---------------------------"

        # viterbi algorithm
        tt4 = time.time()
        for width in xrange(2, length+1):
            for start in xrange(0, length-width+1):
                end = start + width
                # binary rule
                for mid in xrange(start+1, end):
                    for l in cky[start][mid]:
                        for r in cky[mid][end]:
                            time1 = time.time()
                            if (l, r) in self.brules:
                                for p in self.brules[(l, r)]:
                                    if viscore[start][end][p] == 0:
                                        cky[start][end].append(p)
                                    newProb = self.log_prob_binary(lsm, p, l, r, h[mid]) + viscore[start][mid][l] + viscore[mid][end][r]
                                    if newProb > viscore[start][end][p]:
                                        viscore[start][end][p] = newProb
                                        bp[start][end][p] = (mid, l, r)
                            print "binary takes ------ ", time.time() - time1, l, r, p
                print "comes unary"
                # unary rule
                time1 = time.time()
                tmp = []
                for c in cky[start][end]:
                    if c in self.urules:
                        for p in self.urules[c]:
                            print "inside unary ", "p = ", p, " c = ", c
                            if p == c:
                                continue
                            if viscore[start][end][p] == 0:
                                tmp.append(p)
                            newProb = self.log_prob_unt(lsm, p, c, h[start]) + viscore[start][end][c]
                            if newProb > viscore[start][end][p]:
                                viscore[start][end][p] = newProb
                                bp[start][end][p] = (None, None, c)
                cky[start][end] += tmp
                print "binary takes ------ ", time.time() - time1

        tt5 = time.time()
        if self.verbose == 'yes':
            print "Finish inside algorithm ... ", tt5 - tt4

        return bp

    def unsupervised(self, sen):

        emb_inp = self.encoder_t(sen)
        output, hidden = self.LSTM(emb_inp, self.h0)
        sen = sen.view(-1).data
        length = len(sen)
        lsm = nn.LogSoftmax()

        ## pre-compute all probabilities

        h1 = output.repeat(self.nnt, 1, 1)
        h2 = output.unsqueeze(0).repeat(self.nnt, self.nnt, 1, 1)

        unt_i = self.unt_pre.repeat(1, length, 1)
        p2l_i = self.p2l_pre.repeat(1, length, 1)
        pl2r_i = self.pl2r_pre.repeat(1, 1, length, 1)

        unt_cond = torch.cat((unt_i, h1), 2)
        p2l_cond = torch.cat((p2l_i, h1), 2)
        pl2r_cond = torch.cat((pl2r_i, h2), 3)
        size = unt_cond.size()
        size2 = pl2r_cond.size()

        # parent to unary child
        unt_pr = lsm(self.unt(unt_cond.view(-1, size[2]))).view(size[0], size[1], -1)

        # parent to left
        p2l_pr = lsm(self.p2l(p2l_cond.view(-1, size[2]))).view(size[0], size[1], -1)

        # parent left to right
        pl2r_pr = lsm(self.pl2r(p2lr_cond.view(-1, size[3]))).view(size2[0], size2[1], size2[2], -1)

        ## Inside Algorithm

        cky = [[[] for j in xrange(length+1)] for i in xrange(length)]
        viscore = [[[0 for k in xrange(self.nnt)] for j in xrange(length+1)] for i in xrange(length)]

        w2v_w = self.word2vec.weight + self.word2vec_plus.weight
        ut_w = w2v_w.mm(self.ut.weight).t()
        ut_b = w2v_w.mm(self.ut.bias.view(-1, 1)).t()

        # Initialize the chart
        tt0 = time.time()
        for start in xrange(length):
            end = start + 1
            c = sen[start]
            for p in self.lexicon[c]:
                cky[start][end].append(p)
                iscore[start][end][p] = self.log_prob_ut(lsm, ut_w, ut_b, p, c, h[start])

        tt1 = time.time()
        if self.verbose == 'yes':
            print "LEXICON ", tt1-tt0, "---------------------------"

        # Unary appending, deal with non_term -> non_term ... -> term chain
        tt2 = time.time()
        for start in xrange(length):
            end = start + 1
            tmp = []
            for c in cky[start][end]:
                if c in self.urules:
                    for p in self.urules[c]:
                        if iscore[start][end][p] == 0:
                            tmp.append(p)
                        iscore[start][end][p] += self.log_prob_unt(lsm, p, c, h[start]) + iscore[start][end][c]
            cky[start][end] += tmp

        tt3 = time.time()
        if self.verbose == 'yes':
            print "UNARY ", tt3-tt2, "---------------------------"

        # viterbi algorithm
        tt4 = time.time()
        for width in xrange(2, length+1):
            for start in xrange(0, length-width+1):
                end = start + width
                # binary rule
                for mid in xrange(start+1, end):
                    for l in cky[start][mid]:
                        for r in cky[mid][end]:
                            if (l, r) in self.brules:
                                for p in self.brules[(l, r)]:
                                    if iscore[start][end][p] == 0:
                                        cky[start][end].append(p)
                                    iscore[start][end][p] += \
                                        self.log_prob_binary(lsm, p, l, r, h[mid]) + iscore[start][mid][l] + iscore[mid][end][r]

                # unary rule
                for c in cky[start][end]:
                    tmp = []
                    if c in self.urules:
                        for p in self.urules[c]:
                            if iscore[start][end][p] == 0:
                                tmp.append(p)
                            iscore[start][end][p] += \
                                self.log_prob_unt(lsm, p, c, h[start]) + iscore[start][end][c]
                    cky[start][end] += tmp

        tt5 = time.time()
        if self.verbose == 'yes':
            print "Finish inside algorithm ... ", tt5 - tt4

        if cky[0][length][2] == 0:
            if self.cuda_flag:
                return Variable(torch.FloatTensor([-1])).cuda()
            return Variable(torch.FloatTensor([-1]))
        else:
            return -iscore[0][length][2]


    def log_prob_ut(self, lsm, ut_w, ut_b, p, c, h):
        pi = Variable(torch.LongTensor([p]))
        h = h.view(1, -1)
        if self.cuda_flag:
            pi = pi.cuda()
            h = h.cuda()

        cond = torch.cat((self.encoder_nt(pi), h), 1)
        res = cond.mm(ut_w) + ut_b
        return lsm(self.p2l(cond))[0][0] + lsm(res)[0][c] 

    def log_prob_unt(self, lsm, p, c, h):
        pi = Variable(torch.LongTensor([p]))
        h = h.view(1, -1)
        if self.cuda_flag:
            pi = pi.cuda()
            h = h.cuda()

        cond = torch.cat((self.encoder_nt(pi), h), 1)
        return lsm(self.p2l(cond))[0][1] + lsm(self.unt(cond))[0][c]

    def log_prob_binary(self, lsm, p, l, r, h):
        pi = Variable(torch.LongTensor([p]))
        pli = Variable(torch.LongTensor([p, l]))
        h = h.view(1, -1)
        if self.cuda_flag:
            pi = pi.cuda()
            pli = pli.cuda()
            h = h.cuda()

        p2l_cond = torch.cat((self.encoder_nt(pi), h), 1)
        pl2r_cond = torch.cat((self.encoder_nt(pli).view(1,-1), h), 1)
        return lsm(self.p2l(p2l_cond))[0][l] + lsm(self.pl2r(pl2r_cond))[0][r]

    def log_sum_exp(self, a, b):
        m = a if a > b else b
        return m + torch.log(torch.exp(a-m) + torch.exp(b-m))

    def max(self, a, b):
        if a > b:
            return a
        else:
            return b

    def supervised(self, sens,
        p2l, pl2r, unt, ut,
        p2l_t, pl2r_t, unt_t, ut_t,
        p2l_i, pl2r_i, unt_i, ut_i):

        #t0 = time.time()
        # run the LSTM to extract features from left context
        output, hidden = self.LSTM(self.encoder_t(sens), self.h0) 
        output = self.coef_lstm * output.contiguous().view(-1, output.size(2))

        #t1 = time.time()

        logsoftmax = nn.LogSoftmax()

        p2l = torch.cat((
                p2l.view(-1, p2l.size(2)), 
                torch.index_select(output, 0, p2l_i.view(-1)))
            , 1)

        mask = (p2l_t.view(-1) > 0)
        mask_matrix = mask.repeat(1, self.nnt)
        matrix = logsoftmax(self.p2l(p2l)).masked_select(mask_matrix).view(-1, self.nnt)
        target = p2l_t.view(-1).masked_select(mask)
        nll_p2l = -torch.sum( matrix.gather(1, target.unsqueeze(1)) )

        #t2 = time.time()

        pl2r = torch.cat((
                pl2r.view(-1, pl2r.size(2)), 
                torch.index_select(output, 0, pl2r_i.view(-1)))
            , 1)

        mask = (pl2r_t.view(-1) > 0)
        mask_matrix = mask.repeat(1, self.nnt)
        matrix = logsoftmax(self.pl2r(pl2r)).masked_select(mask_matrix).view(-1, self.nnt)
        target = pl2r_t.view(-1).masked_select(mask)
        nll_pl2r = -torch.sum( matrix.gather(1, target.unsqueeze(1)) )

        #t3 = time.time()

        unt = torch.cat((
                unt.view(-1, unt.size(2)), 
                torch.index_select(output, 0, unt_i.view(-1)))
            , 1)

        mask = (unt_t.view(-1) > 0)
        mask_matrix = mask.repeat(1, self.nnt)
        matrix = logsoftmax(self.unt(unt)).masked_select(mask_matrix).view(-1, self.nnt)
        target = unt_t.view(-1).masked_select(mask)
        nll_unt = -torch.sum( matrix.gather(1, target.unsqueeze(1)) )

        #t4 = time.time()

        ut = torch.cat((
                ut.view(-1, ut.size(2)), 
                torch.index_select(output, 0, ut_i.view(-1)))
            , 1)

        mask = (ut_t.view(-1) > 0)
        mask_matrix = mask.repeat(1, self.nt)
        matrix = self.ut(ut).mm((self.word2vec.weight + self.word2vec_plus.weight).t())
        matrix = logsoftmax(matrix).masked_select(mask_matrix).view(-1, self.nt)
        target = ut_t.view(-1).masked_select(mask)
        nll_ut = -torch.sum( matrix.gather(1, target.unsqueeze(1)) )

        #t5 = time.time()

        #print "needs %.4f, %.4f, %.4f, %.4f, %.4f secs" % (round(t1- t0, 5), round(t2- t1, 5), round(t3- t2, 5), round(t4- t3, 5), round(t5-t4, 5))
        nll = nll_p2l + nll_pl2r + nll_unt + nll_ut

        return nll