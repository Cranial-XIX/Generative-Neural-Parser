import copy
import gr
import itertools
import math
import numpy as np
import time
import torch
import torch.nn as nn

from torch.autograd import Variable

"""
The (B)a(S)eline model
"""
class BS(nn.Module):

    def __init__(self, args):
        super(BS, self).__init__()

        self.use_cuda = args['cuda']
        dp = args['dp']                    # the data processor

        # model
        self.bsz = args['bsz']             # the batch size
        self.dhid = args['dhid']           # LSTM hidden dimension size
        self.nlayers = args['nlayers']     # number of layers in neural net

        # terminals
        self.nt = dp.nt                    # number of terminals
        self.dt = dp.dt                    # dimension of terminals

        # nonterminals
        self.nnt = dp.nnt                  # number of nonterminals
        self.dnt = dp.dnt                  # dimension of nonterminals

        self.nunary = dp.nunary            # details please look at processor.py
        self.prefix = dp.unary_prefix
        self.suffix = dp.unary_suffix

        self.B_AC = dp.B_AC
        self.B_A = dp.unary
        self.w_U = dp.w_U
        self.idx2nt = dp.idx2nt

        # nonterminal embedding and w2v embedding, w2v_plus 
        # is the deviation from w2v
        self.nt_emb = nn.Embedding(self.nnt, self.dnt)
        self.word_emb = nn.Embedding(self.nt, self.dt)

        self.lsm = nn.LogSoftmax()
        self.sm = nn.Softmax()
        self.relu = nn.ReLU()

        self.A2B = nn.Linear(self.dnt, self.nnt)
        self.AB2C = nn.Linear(self.dnt*2, self.nnt)
        self.A2U = nn.Linear(self.dnt, self.nunary)
        self.A2T = nn.Linear(self.dnt, self.nt)


    def parse_setup(self):
        n = 30
        self.betas = np.empty((n, n+1, self.nnt, 2))
        self.Bp = np.zeros((n,n+1,self.nnt), dtype=np.int8)
        self.Cp = np.zeros((n,n+1,self.nnt), dtype=int)
        self.jp = np.zeros((n,n+1,self.nnt), dtype=np.int8)

        # Precompute binary rules probabilities
        AA = []
        BB = []
        CC = []
        idx = -1
        self.ABC = {}
        for B in self.B_AC:
            for A, C in self.B_AC[B]:
                idx += 1
                AA.append(A)
                BB.append(B)
                CC.append(C)
                self.ABC[(A,B,C)] = idx

        AA = Variable(torch.LongTensor(AA))
        BB = Variable(torch.LongTensor(BB))
        CC = Variable(torch.LongTensor(CC))

        AAv = self.nt_emb(AA.view(1,-1)).squeeze(0)
        BBv = self.nt_emb(BB.view(1,-1)).squeeze(0)

        nll_B = self.lsm(
            self.A2B(AAv)
        ).gather(1, BB.unsqueeze(1))

        nll_C = self.lsm(
            self.AB2C(
                torch.cat( (AAv, BBv), 1 )
            )
        ).gather(1, CC.unsqueeze(1))

        self.BINARY = (nll_B + nll_C).view(-1).data

        # precompute unary rules probabilities
        AA = []
        UU = []
        idx = -1
        self.AU = {}
        for B in self.B_A:
            for U, A in self.B_A[B]:
                idx += 1
                AA.append(A)
                UU.append(U)
                self.AU[(A,U)] = idx

        AA = Variable(torch.LongTensor(AA))
        UU = Variable(torch.LongTensor(UU))

        AAv = self.nt_emb(AA.view(1,-1)).squeeze(0)

        nll_U = self.lsm(
            self.A2U(AAv)
        ).gather(1, UU.unsqueeze(1))

        self.UNARY = nll_U.view(-1).data


    def parse(self, sentence, sen_idx):
        t0 = time.time()
        self.sentence = sentence.split()
        n = len(sen_idx)-1

        ZERO = -10000
        self.betas.fill(ZERO)

        PP = []
        WW = []

        print sen_idx
        for i in xrange(1,n+1):
            idx = sen_idx[i]
            for U, A, PT in self.w_U[idx]:
                PP.append(PT)
                WW.append(WW)

        PPv = self.nt_emb((Variable(torch.LongTensor(PP))).view(1,-1)).squeeze(0)


        nll_T = -torch.sum(
            self.lsm(
                self.A2T(PPv)
            ).gather(1, (Variable(torch.LongTensor(WW))).unsqueeze(1))
        ).view(-1).data

        index = -1
        for i in xrange(1,n+1):
            idx = sen_idx[i]
            for U, A, PT in self.w_U[idx]:
                index += 1
                score = nll_T[index] + self.UNARY[self.AU[A, U]]
                if score > self.betas[i-1,i,A,1]:
                    self.betas[i-1,i,A,1] = score
                    self.Bp[i-1,i,A] = -1
                    self.Cp[i-1,i,A] = U

        t1 = time.time()
        #print " From start to initialization : ", t1 - t0

        '''
        For the CKY algorithm, for each layer of spans of certain width,
        we collect all we need to compute, then we compute them all to save
        time. Also, by doing this, we reduce the computation for next layer above
        if we are sure certain spans are not possible (in viterbi algorithm, this saves
        time)
        '''
        for w in xrange(2, n+1):

            for i in xrange(n-w+1):
                k = i + w
                for j in xrange(i+1, k):
                    for B in self.B_AC:
                        Bs = self.betas[i,j,B,1]
                        if Bs == ZERO:
                            continue
                        for A, C in self.B_AC[B]:
                            Cs = self.betas[j,k,C,1]
                            if Cs == ZERO:
                                continue
                            score = self.BINARY[self.ABC[(A,B,C)]] + Bs + Cs
                            if score > self.betas[i,k,A,0]:
                                # print "Binary: ({}, {}, {}) {} ({})-> {} {} ({}) = {}".format(i,j,k, self.idx2nt[A], Bh, self.idx2nt[B], self.idx2nt[C], Ch, ikAB)
                                self.betas[i,k,A,0] = score
                                self.Bp[i,k,A] = B
                                self.Cp[i,k,A] = C
                                self.jp[i,k,A] = j

            for i in xrange(n-w+1):
                k = i + w
                for A in xrange(self.nnt):
                    As = self.betas[i,k,A,0]
                    if As > ZERO:
                        self.betas[i,k,A,1] = As

            for i in xrange(n-w+1):
                k = i + w
                for B in self.B_A:
                    Bs = self.betas[i,k,B,0]
                    if Bs == ZERO:
                        continue
                    for U, A in self.B_A[B]:
                        index += 1
                        score = self.UNARY[self.AU[(A, U)]] + Bs
                        if score > self.betas[i,k,A,1]:
                            self.betas[i,k,A,1] = score
                            self.Bp[i,k,A] = B
                            self.Cp[i,k,A] = U
                            self.jp[i,k,A] = -1

        return self.print_parse(0, n, 1)


    def print_parse(self, i, k, A):

        B = self.Bp[i,k,A]
        C = self.Cp[i,k,A]
        j = self.jp[i,k,A]

        if B == -1:
            # is terminal rule
            return self.prefix[C] + " " + self.sentence[i] + self.suffix[C]
        elif j == -1:
            # unary rule
            return self.prefix[C] + self.print_parse(i, k, B) + self.suffix[C]
        else:
            # binary rule
            #print i, j, k, self.idx2nt[A], self.idx2nt[B], self.idx2nt[C]
            return  "(" + self.idx2nt[A] + " " \
                + self.print_parse(i, j, B) + " " \
                + self.print_parse(j, k, C) + ")"        

    def forward(self, train_type, args):
        if train_type == 'supervised':
            return self.supervised(*args)
        elif train_type == 'unsupervised':
            return self.unsupervised(*args)
        else:
            print "Unrecognized train type!"
            return

    def supervised(self, AA, BB, CC, U, U_A, T, T_A):

        nll_B = -torch.sum(
            self.lsm(
                self.A2B(self.nt_emb(AA))
            ).gather(1, BB.unsqueeze(1))
        )

        nll_C = -torch.sum(
            self.lsm(
                self.AB2C(
                    torch.cat( (self.nt_emb(AA), self.nt_emb(BB)), 1 )
                )
            ).gather(1, CC.unsqueeze(1))
        )

        nll_U = -torch.sum(
            self.lsm(
                self.A2U(
                    self.nt_emb(U_A)
                )
            ).gather(1, U.unsqueeze(1))
        )

        nll_T = -torch.sum(
            self.lsm(
                self.A2T(
                    self.nt_emb(T_A)
                )
            ).gather(1, T.unsqueeze(1))
        )

        return nll_B + nll_C + nll_U + nll_T

"""
The (B)a(S)eline model with (N)eural network
"""
class BSN(nn.Module):

    def __init__(self, args):
        super(BSN, self).__init__()

"""
The model using (L)eft context lstm features and (N)eural network
"""
class LN(nn.Module):

    def __init__(self, args):
        super(LN, self).__init__()

        self.verbose = args['verbose']
        self.use_cuda = args['cuda']

        # terminals
        self.nt = args['nt']               # number of terminals
        self.dt = args['dt']               # dimension of terminals
        self.dt = 100

        # nonterminals
        self.nnt = args['nnt']             # number of nonterminals
        self.dnt = args['dnt']             # dimension of nonterminals
        self.dnt = 30

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
            batch_first=True, bias=True, dropout=0.6
        )

        dp2l = dunt = dut = self.dnt + self.dhid
        dpl2r = 2 * (self.dnt + self.dhid)

        self.lsm = nn.LogSoftmax()
        self.sm = nn.Softmax()
        self.relu = nn.ReLU()

        zeta = 0.7

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
        self.ut = nn.Linear(dut, 200)
        self.ut_out = nn.Linear(200, self.nt)
        self.init_weights(args['initrange'], args['word_emb'], args['nt_emb'])

    def init_weights(self, initrange, word_emb, nt_emb):
        #self.word_emb.weight.data = word_emb
        #self.nt_emb.weight.data = nt_emb
        #self.word_emb.weight.requires_grad = False
        #self.nt_emb.weight.requires_grad = False

        # Below are initial setup for LSTM
        lstm_weight_range = 0.01

        self.LSTM.weight_ih_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l0.data.uniform_(-lstm_weight_range, lstm_weight_range)

        self.LSTM.weight_ih_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)
        self.LSTM.weight_hh_l1.data.uniform_(-lstm_weight_range, lstm_weight_range)

        self.LSTM.weight_ih_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)  
        self.LSTM.weight_hh_l2.data.uniform_(-lstm_weight_range, lstm_weight_range)

        size = len(self.LSTM.bias_ih_l0)
        section = size / 4
        for i in xrange(section, 2*section):
            self.LSTM.bias_ih_l0.data[i] = 1.0
            self.LSTM.bias_hh_l0.data[i] = 1.0

            self.LSTM.bias_ih_l1.data[i] = 1.0
            self.LSTM.bias_hh_l1.data[i] = 1.0

            self.LSTM.bias_ih_l2.data[i] = 1.0
            self.LSTM.bias_hh_l2.data[i] = 1.0

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
        self.ut_out.bias.data.fill_(0)
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

    def parse(self, sentence, sen, pos=None, viterbi=True):
        # sen is a torch.LongTensor object, containing fake BOS index
        # along with indices of the words

        t0 = time.time()
        self.pos_correct = 0
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
            if idx == pos[i]:
                self.pos_correct += 1
            self.pos.append(idx)

        '''
        for i in xrange(n):
            c = sen[i]
            p = pos[i]
            pi = Variable(torch.LongTensor([p]))
            h = output[i].view(1, -1)
            if self.use_cuda:
                pi = pi.cuda()
                h = h.cuda()
            cond = torch.cat((self.nt_emb(pi), h), 1)
            #res = cond.mm(self.ut_w) + self.ut_b
            res = self.ut_out(self.relu(self.ut(cond)))

            preterminal[i,p] = (softmax(self.p2l(cond))[0][U_TM] + softmax(res)[0][c]).data[0]
        '''
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
                        unt_pr.self.Cpu().data.numpy(),
                        p2l_pr.self.Cpu().data.numpy(),
                        pl2r_pr.self.Cpu().data.numpy()
                )
            else:
                return self.parser.mbr(
                        sentence,
                        preterminal,
                        unt_pr.self.Cpu().data.numpy(),
                        p2l_pr.self.Cpu().data.numpy(),
                        pl2r_pr.self.Cpu().data.numpy()
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

"""
The model with (B)ilexical information, (L)eft context LSTM features,
and (N)eural network.
"""
class BLN(nn.Module):

    def __init__(self, args):
        super(BLN, self).__init__()

        self.use_cuda = args['cuda']
        dp = args['dp']                    # the data processor

        # model
        self.bsz = args['bsz']             # the batch size
        self.dhid = args['dhid']           # LSTM hidden dimension size
        self.nlayers = args['nlayers']     # number of layers in neural net

        # terminals
        self.nt = dp.nt                    # number of terminals
        self.dt = dp.dt                    # dimension of terminals

        # nonterminals
        self.nnt = dp.nnt                  # number of nonterminals
        self.dnt = dp.dnt                  # dimension of nonterminals

        self.nunary = dp.nunary            # details please look at processor.py
        self.prefix = dp.unary_prefix
        self.suffix = dp.unary_suffix

        self.B_AC = dp.B_AC
        self.B_A = dp.unary
        self.w_U = dp.w_U

        self.idx2nt = dp.idx2nt

        self.init_h0()
        self.h0[0].requires_grad = False
        self.h0[1].requires_grad = False

        # nonterminal embedding and w2v embedding, w2v_plus 
        # is the deviation from w2v
        self.nt_emb = nn.Embedding(self.nnt, self.dnt)
        self.word_emb = nn.Embedding(self.nt, self.dt)

        # The LSTM and some linear transformation layers
        self.LSTM = nn.LSTM(
            self.dt, self.dhid, self.nlayers,
            batch_first=True, bias=True, dropout=0.6
        )

        init = 0.01

        self.LSTM.weight_ih_l0.data.uniform_(-init, init)
        self.LSTM.weight_hh_l0.data.uniform_(-init, init)

        self.LSTM.weight_ih_l1.data.uniform_(-init, init)
        self.LSTM.weight_hh_l1.data.uniform_(-init, init)

        self.LSTM.weight_ih_l2.data.uniform_(-init, init)  
        self.LSTM.weight_hh_l2.data.uniform_(-init, init)

        self.lsm = nn.LogSoftmax()
        self.sm = nn.Softmax()
        self.relu = nn.ReLU()

        # below are linear layers in single layer neural net
        B_in = self.dt + self.dnt + self.dhid
        B_out = self.nnt * 2

        C_in = self.dt + self.dnt * 2 + self.dhid * 2
        C_out = self.nnt

        h_in = self.dt + self.dnt + self.dhid
        h_out = self.nt

        h1_in = self.dt + self.dnt + self.dhid
        h1_out = self.nt

        C1_in = self.dt * 2 + self.dnt * 2 + self.dhid * 2
        C1_out = self.nnt

        U_in = self.dt + self.dnt + self.dhid
        U_out = self.nunary

        zeta = 0.6
        d_B = int( zeta * B_out + (1-zeta) * B_in )
        d_C = int( zeta * C_out + (1-zeta) * C_in )
        d_h = int( h_in )
        d_h1 = int( h1_in )
        d_C1 = int( zeta * C1_out + (1-zeta) * C1_in )
        d_U = int( zeta * U_out + (1-zeta) * U_in )

        self.B_h1 = nn.Linear(B_in, d_B)
        self.B_h2 = nn.Linear(d_B, B_out)
        self.B_h1.weight.data.uniform_(-init, init)
        self.B_h2.weight.data.uniform_(-init, init)

        self.C_h1 = nn.Linear(C_in, d_C)
        self.C_h2 = nn.Linear(d_C, C_out)
        self.C_h1.weight.data.uniform_(-init, init)
        self.C_h2.weight.data.uniform_(-init, init)

        self.h_h1 = nn.Linear(h_in, d_h)
        self.h_h2 = nn.Linear(d_h, h_out)
        self.h_h1.weight.data.uniform_(-init, init)
        self.h_h2.weight.data.uniform_(-init, init)

        self.h1_h1 = nn.Linear(h1_in, d_h1)
        self.h1_h2 = nn.Linear(d_h1, h1_out)
        self.h1_h1.weight.data.uniform_(-init, init)
        self.h1_h2.weight.data.uniform_(-init, init)

        self.C1_h1 = nn.Linear(C1_in, d_C1)
        self.C1_h2 = nn.Linear(d_C1, C1_out)
        self.C1_h1.weight.data.uniform_(-init, init)
        self.C1_h2.weight.data.uniform_(-init, init)

        self.U_h1 = nn.Linear(U_in, d_U)
        self.U_h2 = nn.Linear(d_U, U_out)
        self.U_h1.weight.data.uniform_(-init, init)
        self.U_h2.weight.data.uniform_(-init, init)


    def init_h0(self, bsz=None):
        if bsz == None:
            bsz = self.bsz
        # the initial h for the LSTM
        if self.use_cuda:
            # the initial states for h0 and c0 of LSTM
            self.h0 = (
                Variable(torch.zeros(self.nlayers, bsz, self.dhid).cuda()),
                Variable(torch.zeros(self.nlayers, bsz, self.dhid).cuda())
            )
        else:
            self.h0 = (
                Variable(torch.zeros(self.nlayers, bsz, self.dhid)),
                Variable(torch.zeros(self.nlayers, bsz, self.dhid))
            )

    def parse_setup(self):
        n = 30
        self.init_h0(1)
        self.betas = np.empty((n, n+1, self.nnt, 2))
        self.Bp = np.zeros((n,n+1,self.nnt), dtype=np.int8)
        self.Cp = np.zeros((n,n+1,self.nnt), dtype=int)
        self.jp = np.zeros((n,n+1,self.nnt), dtype=np.int8)
        self.H = np.zeros((n,n+1,self.nnt), dtype=int)

    def parse(self, sentence, sen_idx):
        t0 = time.time()
        self.sentence = sentence.split()
        V = self.word_emb(sen_idx)
        sen_idx = sen_idx.data
        alpha, _ = self.LSTM(V.unsqueeze(0), self.h0)
        alpha = alpha.squeeze(0)[:-1]
        n = len(sen_idx)-1

        ZERO = -10000
        self.betas.fill(ZERO)

        '''
        # This commented out part is the original implementation when we
        # allow all unary chain on top of a terminal symbol. We changed it
        # to only allow unary chains that are compatible with the word.

        # P( unary | A, alpha(A), v(h))
        Us, As = zip(*self.B_A[0])
        As_var = Variable(torch.LongTensor(As))
        x = len(As)
        U = Variable(torch.LongTensor(Us))

        init = self.lsm(
            self.U_h2(
                self.relu(
                    self.U_h1(
                        torch.cat((
                            self.nt_emb(As_var).unsqueeze(0).repeat(n, 1, 1),
                            alpha.unsqueeze(1).repeat(1, x, 1),
                            V[1:].unsqueeze(1).repeat(1, x, 1),
                        ), 2).view(n*x, -1)
                    )
                )
            )
        ).gather(1, U.repeat(n).unsqueeze(1)).view(n,x).data

        # Do inside algorithm
        ZERO = -10000
        self.betas.fill(ZERO)

        # Initialization
        # append one level of unary chain above the terminals
        for i in xrange(n):
            curr = init[i]
            #threshold = torch.max(curr).data[0] - 4
            for a in xrange(x):
                score = curr[a]
                nt = As[a]
                if score > self.betas[i,i+1,nt,1]:
                    self.betas[i,i+1,nt,1] = score
                    self.Bp[i,i+1,nt] = -1
                    self.Cp[i,i+1,nt] = Us[a]
                    self.jp[i,i+1,nt] = i
                    self.H[i,i+1,nt] = sen_idx[i+1]       # +1 for BOS
        '''
        AA = []
        UU = []
        Ai = []
        HH = []

        for i in xrange(1,n+1):
            idx = sen_idx[i]
            for U, A, PT in self.w_U[idx]:
                AA.append(A)
                UU.append(U)
                Ai.append(i-1)
                HH.append(idx)

        Aiv = torch.index_select(alpha, 0, Variable(torch.LongTensor(Ai)))
        AAv = self.nt_emb((Variable(torch.LongTensor(AA))).view(1,-1)).squeeze(0)
        AHv = self.word_emb(Variable(torch.LongTensor(HH)))

        nll_U = self.lsm(
            self.U_h2(
                self.relu(
                    self.U_h1(
                        torch.cat( (AAv, Aiv, AHv), 1 )
                    )
                )
            )
        ).gather(1, (Variable(torch.LongTensor(UU))).unsqueeze(1)).squeeze(1).data


        index = -1
        for i in xrange(1,n+1):
            idx = sen_idx[i]
            #threshold = torch.max(curr).data[0] - 4
            for U, A, PT in self.w_U[idx]:
                index += 1
                score = nll_U[index]
                if score > self.betas[i-1,i,A,1]:
                    self.betas[i-1,i,A,1] = score
                    self.Bp[i-1,i,A] = -1
                    self.Cp[i-1,i,A] = U
                    self.H[i-1,i,A] = idx      # +1 for BOS    

        t1 = time.time()
        #print " From start to initialization : ", t1 - t0

        '''
        For the CKY algorithm, for each layer of spans of certain width,
        we collect all we need to compute, then we compute them all to save
        time. Also, by doing this, we reduce the computation for next layer above
        if we are sure certain spans are not possible (in viterbi algorithm, this saves
        time)
        '''
        for w in xrange(2, n+1):

            t2 = time.time()
            AA = []
            BB = []
            CC = []
            Ai = []
            Ci = []
            BH = []
            CH = []

            BH_d = {}
            BH_n = -1
            BH_C = []
            BH_BH = []
            BH_Ci = []

            CH_d = {}
            CH_n = -1
            CH_B = []
            CH_CH = []
            CH_Bi = []


            tt0 = time.time()
            for i in xrange(n-w+1):
                k = i + w
                for j in xrange(i+1, k):
                    for B in self.B_AC:
                        Bs = self.betas[i,j,B,1]
                        if Bs == ZERO:
                            continue
                        Bh = self.H[i,j,B]
                        for A, C in self.B_AC[B]:
                            Cs = self.betas[j,k,C,1]
                            if Cs == ZERO:
                                continue
                            Ch = self.H[j,k,C]

                            AA.append(A)
                            BB.append(B)
                            CC.append(C)
                            Ai.append(i)
                            Ci.append(j)
                            BH.append(Bh)
                            CH.append(Ch)

                            tpl1 = (C, Bh, j)
                            tpl2 = (B, Ch, i)
                            if tpl1 not in BH_d:
                                BH_n += 1
                                BH_d[tpl1] = BH_n
                                BH_C.append(C)
                                BH_BH.append(Bh)
                                BH_Ci.append(j)
                            if tpl2 not in CH_d:
                                CH_n += 1
                                CH_d[tpl2] = CH_n
                                CH_B.append(B)
                                CH_CH.append(Ch)
                                CH_Bi.append(i)

            if not AA:
                continue

            t3 = time.time()

            #print " LEN of AA is : ", len(AA)


            XB = Variable(torch.LongTensor(BB))
            XC = Variable(torch.LongTensor(CC))
            Aiv = torch.index_select(alpha, 0, Variable(torch.LongTensor(Ai)))
            Civ = torch.index_select(alpha, 0, Variable(torch.LongTensor(Ci)))
            AAv = self.nt_emb(Variable(torch.LongTensor(AA)).view(1,-1)).squeeze(0)
            BBv = self.nt_emb(XB.view(1,-1)).squeeze(0)
            CCv = self.nt_emb(XC.view(1,-1)).squeeze(0)
            BHv = self.word_emb(Variable(torch.LongTensor(BH)))
            CHv = self.word_emb(Variable(torch.LongTensor(CH)))

            BH_Cv = self.nt_emb(Variable(torch.LongTensor(BH_C)).view(1,-1)).squeeze(0)
            BH_BHv = self.word_emb(Variable(torch.LongTensor(BH_BH)))
            BH_Civ = torch.index_select(alpha, 0, Variable(torch.LongTensor(BH_Ci)))

            CH_Bv = self.nt_emb(Variable(torch.LongTensor(CH_B)).view(1,-1)).squeeze(0)
            CH_CHv = self.word_emb(Variable(torch.LongTensor(CH_CH)))
            CH_Biv = torch.index_select(alpha, 0, Variable(torch.LongTensor(CH_Bi)))

            t4 = time.time()

            term = 0

            # Below is when B is the head

            # P( @B/B | A, v(h), alpha(A) )
            nll_B1 = self.lsm(
                self.B_h2(
                    self.relu(
                        self.B_h1(
                            torch.cat( (AAv, BHv, Aiv), 1 )
                        )
                    )
                )
            ).gather(1, (XB+self.nnt).unsqueeze(1))

            # P( C | A, B, v(h), alpha(A), alpha(C)-alpha(A) )
            nll_C = self.lsm(
                self.C_h2(
                    self.relu(
                        self.C_h1(
                            torch.cat( (AAv, BBv, BHv, Aiv, Civ-Aiv), 1 )
                        )
                    )
                )
            ).gather(1, XC.unsqueeze(1))

            ss = time.time()
            # P( h' | C, BH, Ci)
            nll_h = self.lsm(
                self.h_h2(
                    self.relu(
                        self.h_h1(
                            torch.cat( (BH_Cv, BH_BHv, BH_Civ), 1 )
                        )
                    )
                )
            ).data
            term += time.time() - ss

            B_PR = (nll_B1 + nll_C).squeeze(1).data


            # Below is when C is head

            nll_B2 = self.lsm(
                self.B_h2(
                    self.relu(
                        self.B_h1(
                            torch.cat( (AAv, CHv, Aiv), 1 )
                        )
                    )
                )
            ).gather(1, XB.unsqueeze(1))


            ss = time.time()
            # P( h' | A, B, v(h), alpha(A) )
            nll_h1 = self.lsm(
                self.h1_h2(
                    self.relu(
                        self.h1_h1(
                            torch.cat( (CH_Bv, CH_CHv, CH_Biv), 1 )
                        )
                    )
                )
            ).data
            term += time.time() - ss

            # P( C | A, B, v(h), v(h'), alpha(A), alpha(C) )

            nll_C1 = self.lsm(
                self.C1_h2(
                    self.relu(
                        self.C1_h1(
                            torch.cat((AAv, BBv, BHv, CHv, Aiv, Civ-Aiv), 1))
                    )
                )
            ).gather(1, XC.unsqueeze(1))

            C_PR = (nll_B2 + nll_C1).squeeze(1).data

            t5 = time.time()

            #print " GPU : ", t5-t4
            #print " MEM : ", t4 -t3
            #print " TERM : {} %".format(term * 100 / (t5-t4)) 
            index = -1
            for i in xrange(n-w+1):
                k = i + w
                for j in xrange(i+1, k):
                    for B in self.B_AC:
                        Bs = self.betas[i,j,B,1]
                        if Bs == ZERO:
                            continue
                        Bh = self.H[i,j,B]
                        for A, C in self.B_AC[B]:
                            Cs = self.betas[j,k,C,1]
                            if Cs == ZERO:
                                continue
                            Ch = self.H[j,k,C]
                            index += 1
                            ikAB = B_PR[index] + Bs + Cs + nll_h[BH_d[(C, Bh, j)]][Ch]
                            ikAC = C_PR[index] + Bs + Cs + nll_h1[CH_d[(B, Ch, i)]][Bh]

                            if ikAB >= ikAC:
                                if ikAB > self.betas[i,k,A,0]:
                                    # print "Binary: ({}, {}, {}) {} ({})-> {} {} ({}) = {}".format(i,j,k, self.idx2nt[A], Bh, self.idx2nt[B], self.idx2nt[C], Ch, ikAB)
                                    self.betas[i,k,A,0] = ikAB
                                    self.H[i,k,A] = Bh
                                    self.Bp[i,k,A] = B
                                    self.Cp[i,k,A] = C
                                    self.jp[i,k,A] = j
                            elif ikAC > self.betas[i,k,A,0]:
                                    #print "Binary: ({}, {}, {}) {} ({})-> {} ({}) {} = {}".format(i,j,k, self.idx2nt[A], Ch, self.idx2nt[B], Bh, self.idx2nt[C], ikAB)
                                    self.betas[i,k,A,0] = ikAC
                                    self.H[i,k,A] = Ch
                                    self.Bp[i,k,A] = B
                                    self.Cp[i,k,A] = C
                                    self.jp[i,k,A] = j

            t6 = time.time()

            #print " Original: {:4f} B: {:4f} C: {:4f} H: {:4f} H': {:4f} C1: {:4f} Final: {:4f}".format(tt1-tt0, tt2-tt1, tt3-tt2, tt4-tt3, tt5-tt4, tt6-tt5, t3-tt6)
            #print " B : ", t3 - t2

            for i in xrange(n-w+1):
                k = i + w
                for A in xrange(self.nnt):
                    As = self.betas[i,k,A,0]
                    if As > ZERO:
                        self.betas[i,k,A,1] = As

            AA = []
            UU = []
            Ai = []
            HH = []
            for i in xrange(n-w+1):
                k = i + w
                for B in self.B_A:

                    if self.betas[i,k,B,0] == ZERO:
                        continue

                    head = self.H[i,k,B]

                    for U, A in self.B_A[B]:
                        AA.append(A)
                        UU.append(U)
                        Ai.append(i)
                        HH.append(head)

            if not AA:
                continue

            Aiv = torch.index_select(alpha, 0, Variable(torch.LongTensor(Ai)))
            AAv = self.nt_emb((Variable(torch.LongTensor(AA))).view(1,-1)).squeeze(0)
            AHv = self.word_emb(Variable(torch.LongTensor(HH)))

            nll_U = self.lsm(
                self.U_h2(
                    self.relu(
                        self.U_h1(
                            torch.cat( (AAv, Aiv, AHv), 1 )
                        )
                    )
                )
            ).gather(1, (Variable(torch.LongTensor(UU))).unsqueeze(1)).squeeze(1).data


            index = -1
            for i in xrange(n-w+1):
                k = i + w
                for B in self.B_A:
                    Bs = self.betas[i,k,B,0]
                    if Bs == ZERO:
                        continue
                    head = self.H[i,k,B]
                    for U, A in self.B_A[B]:
                        index += 1
                        score = nll_U[index] + Bs
                        if score > self.betas[i,k,A,1]:
                            self.betas[i,k,A,1] = score
                            self.H[i,k,A] = head
                            self.Bp[i,k,A] = B
                            self.Cp[i,k,A] = U
                            self.jp[i,k,A] = -1

            t7 = time.time()

        #print " CKY takes : ", t7 - t0

        return self.print_parse(0, n, 1)


    def print_parse(self, i, k, A):

        B = self.Bp[i,k,A]
        C = self.Cp[i,k,A]
        j = self.jp[i,k,A]

        if B == -1:
            # is terminal rule
            return self.prefix[C] + " " + self.sentence[i] + self.suffix[C]
        elif j == -1:
            # unary rule
            return self.prefix[C] + self.print_parse(i, k, B) + self.suffix[C]
        else:
            # binary rule
            #print i, j, k, self.idx2nt[A], self.idx2nt[B], self.idx2nt[C]
            return  "(" + self.idx2nt[A] + " " \
                + self.print_parse(i, j, B) + " " \
                + self.print_parse(j, k, C) + ")"        

    def forward(self, train_type, args):
        if train_type == 'supervised':
            return self.supervised(*args)
        elif train_type == 'unsupervised':
            return self.unsupervised(*args)
        else:
            print "Unrecognized train type!"
            return

    def supervised(self, sens,
        B_A, B_B, B_C, B_BH, B_CH, B_Bi, B_Ci,
        C_A, C_B, C_C, C_BH, C_CH, C_Bi, C_Ci,
        U, U_A, U_Ai, U_H):

        '''
        compute the NLL of the current batch of training input and return it.
        NLL = NLL_binary_rules + NLL_unary_rules

        NLL_binary_rules = -log( P( B or @B | conditions) * P(C | conditions) 
            * P( new head of nonhead child | conditions) )
        '''

        #t0 = time.time()

        # run the LSTM to extract features from left context
        output, _ = self.LSTM(self.word_emb(sens), self.h0)
        output = output.contiguous().view(-1, output.size(2))


        # B is head
        # P( @B | A, v(h), alpha(A) )
        # P( C | A, B, v(h), alpha(A), alpha(C) )
        # P( h' | C, v(h), alpha(C) )

        Bi = torch.index_select(output, 0, B_Bi)
        Ci = torch.index_select(output, 0, B_Ci)
        A = self.nt_emb(B_A)
        B = self.nt_emb(B_B)
        C = self.nt_emb(B_C)
        BH = self.word_emb(B_BH)

        nll_atB = -torch.sum(
            self.lsm(
                self.B_h2(
                    self.relu(
                        self.B_h1(
                            torch.cat(
                                (A, BH, Bi), 1
                            )
                        )
                    )
                )
            ).gather(1, (B_B+self.nnt).unsqueeze(1))
        )

        #t1 = time.time()

        nll_C = -torch.sum(
            self.lsm(
                self.C_h2(
                    self.relu(
                        self.C_h1(
                            torch.cat( (A, B, BH, Bi, Ci-Bi), 1 )
                        )
                    )
                )
            ).gather(1, B_C.unsqueeze(1))
        )

        #t2 = time.time()

        nll_h = -torch.sum(
            self.lsm(
                self.h_h2(
                    self.relu(
                        self.h_h1(
                            torch.cat( (C, BH, Ci), 1 )
                        )
                    )
                )
            ).gather(1, B_CH.unsqueeze(1))
        )

        #t3 = time.time()


        # C is head, @C
        # P( h' | B, v(h), alpha(A) )
        # P( C | A, B, v(h), v(h'), alpha(A), alpha(C) )
        Bi = torch.index_select(output, 0, C_Bi)
        Ci = torch.index_select(output, 0, C_Ci)
        A = self.nt_emb(C_A)
        B = self.nt_emb(C_B)
        C = self.nt_emb(C_C)
        BH = self.word_emb(C_BH)
        CH = self.word_emb(C_CH)

        nll_B = -torch.sum(
            self.lsm(
                self.B_h2(
                    self.relu(
                        self.B_h1(
                            torch.cat(
                                (A, CH, Bi), 1
                            )
                        )
                    )
                )
            ).gather(1, C_B.unsqueeze(1))
        )

        #t4 = time.time()

        nll_h1 = -torch.sum(
            self.lsm(
                self.h1_h2(
                    self.relu(
                        self.h1_h1(
                            torch.cat( (B, CH, Bi), 1 )
                        )
                    )
                )
            ).gather(1, C_BH.unsqueeze(1))
        )

        #t5 = time.time()

        nll_C1 = -torch.sum(
            self.lsm(
                self.C1_h2(
                    self.relu(
                        self.C1_h1(
                            torch.cat( (A, B, BH, CH, Bi, Ci-Bi), 1 )
                        )
                    )
                )
            ).gather(1, C_C.unsqueeze(1))
        )

        #t6 = time.time()

        # P( unary | A, alpha(A), v(h))
        nll_U = -torch.sum(
            self.lsm(
                self.U_h2(
                    self.relu(
                        self.U_h1(
                            torch.cat((
                                self.nt_emb(U_A),
                                torch.index_select(output, 0, U_Ai),
                                self.word_emb(U_H)
                            ), 1)
                        )
                    )
                )
            ).gather(1, U.unsqueeze(1))
        )
        #t7 = time.time()
        #tot = (t7-t0)/100
        #print "atB {:.2f}% C {:.2f}% h {:.2f}% B {:.2f}% h1 {:.2f}% C1 {:.2f}% U {:.2f}%".format((t1-t0)/tot,(t2-t1)/tot,(t3-t2)/tot,(t4-t3)/tot,(t5-t4)/tot,(t6-t5)/tot, (t7-t6)/tot)
        return nll_atB + nll_C + nll_h + nll_B + nll_h1 + nll_C1 + nll_U


    def check(self, sens, 
        B_A, B_B, B_C, B_BH, B_CH, B_Bi, B_Ci,
        C_A, C_B, C_C, C_BH, C_CH, C_Bi, C_Ci,
        U, U_A, U_Ai, U_H):
        '''
        Helper function to check if our computation in supervised learning is correct.
        '''

        # run the LSTM to extract features from left context
        output, _ = self.LSTM(self.word_emb(sens), self.h0)
        output = output.contiguous().view(-1, output.size(2))

        # B is head
        # P( @B | A, v(h), alpha(A) )
        # P( C | A, B, v(h), alpha(A), alpha(C) )
        # P( h' | C, v(h), alpha(C) )

        Bi = torch.index_select(output, 0, B_Bi)
        Ci = torch.index_select(output, 0, B_Ci)
        A = self.nt_emb(B_A)
        B = self.nt_emb(B_B)
        C = self.nt_emb(B_C)
        BH = self.word_emb(B_BH)

        smatB = self.sm(
            self.B_h2(
                self.relu(
                    self.B_h1(
                        torch.cat(
                            (A, BH, Bi), 1
                        )
                    )
                )
            )
        )

        #t1 = time.time()

        smC = self.sm(
            self.C_h2(
                self.relu(
                    self.C_h1(
                        torch.cat( (A, B, BH, Bi, Ci-Bi), 1 )
                    )
                )
            )
        )

        #t2 = time.time()

        smH = self.sm(
            self.h_h2(
                self.relu(
                    self.h_h1(
                        torch.cat( (C, BH, Ci), 1 )
                    )
                )
            )
        )

        #t3 = time.time()


        # C is head, @C
        # P( h' | B, v(h), alpha(A) )
        # P( C | A, B, v(h), v(h'), alpha(A), alpha(C) )
        Bi = torch.index_select(output, 0, C_Bi)
        Ci = torch.index_select(output, 0, C_Ci)
        A = self.nt_emb(C_A)
        B = self.nt_emb(C_B)
        C = self.nt_emb(C_C)
        BH = self.word_emb(C_BH)
        CH = self.word_emb(C_CH)

        smB = self.sm(
            self.B_h2(
                self.relu(
                    self.B_h1(
                        torch.cat(
                            (A, CH, Bi), 1
                        )
                    )
                )
            )
        )

        #t4 = time.time()

        smH1 = self.sm(
            self.h1_h2(
                self.relu(
                    self.h1_h1(
                        torch.cat( (B, CH, Bi), 1 )
                    )
                )
            )
        )

        #t5 = time.time()

        smC1 = self.sm(
            self.C1_h2(
                self.relu(
                    self.C1_h1(
                        torch.cat( (A, B, BH, CH, Bi, Ci-Bi), 1 )
                    )
                )
            )
        )

        #t6 = time.time()

        # P( unary | A, alpha(A), v(h))
        smU = self.sm(
            self.U_h2(
                self.relu(
                    self.U_h1(
                        torch.cat((
                            self.nt_emb(U_A),
                            torch.index_select(output, 0, U_Ai),
                            self.word_emb(U_H)
                        ), 1)
                    )
                )
            )
        )

        return smatB, smC, smH, smB, smH1, smC1, smU
