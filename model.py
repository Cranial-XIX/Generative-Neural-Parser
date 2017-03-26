import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

class LCNPModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, inputs):
        super(LCNPModel, self).__init__()

        # read in necessary inputs
        initrange = inputs['initrange']

        # terminals
        self.term_emb = inputs['term_emb']
        self.nt = inputs['nt']              # number of terminals
        self.dt = inputs['dt']              # dimension of terminals

        # nonterminals
        self.encoder_nt = inputs['nt_emb']
        self.nnt = inputs['nnt']            # number of nonterminals
        self.dnt = inputs['dnt']            # dimension of nonterminals

        # model
        self.coef_lstm = inputs['coef_lstm']# coefficient of LSTM
        self.bsz = inputs['bsz']
        self.dhid = inputs['dhid']          # dimension of hidden layer
        self.nlayers = inputs['nlayers']    # number of layers in neural net
        self.coef_l2 = inputs['coef_l2']    # coefficient for L2
        initrange = inputs['initrange']
        self.urules = inputs['urules']      # dictionary of unary rules
        self.brules = inputs['brules']      # dictionary of binary rules
        self.lexicon = inputs['lexicon']


        self.encoder_t = nn.Embedding(self.nt, self.dt)

        self.LSTM = nn.LSTM(self.dt, self.dhid, self.nlayers, batch_first=True, bias=True)
        # the initial states for h0 and c0 of LSTM
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
        self.encoder_t.weight.data = self.term_emb.clone()

        self.p2l.bias.data.fill_(0)
        self.p2l.weight.data.uniform_(-initrange, initrange)

        self.pl2r.bias.data.fill_(0)
        self.pl2r.weight.data.uniform_(-initrange, initrange)

        self.unt.bias.data.fill_(0)
        self.unt.weight.data.uniform_(-initrange, initrange)

        self.ut.bias.data.fill_(0)
        self.ut.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, seq_term, seq_preterm=None, 
        p2l=None, p2l_target=None, 
        pl2r=None, pl2r_target=None,
        unt=None, unt_target=None):
        if type(seq_preterm) == Variable:
            return self.supervised(seq_term, seq_preterm, 
                p2l, p2l_target, pl2r, pl2r_target, unt, unt_target)
        else:
            return self.unsupervised(seq_term)

    def parse(self, seq_term):
        emb_inp = self.encoder_t(seq_term)
        output, hidden = self.coef_lstm * self.LSTM(emb_inp, self.h0)       
        
        nll = Variable(torch.FloatTensor([0]))

        sen = seq_term[0]
        left_context = output[0]

        length = len(sen)
        # every entry is a list of tuples, with each tuple indicate a potential nonterminal 
        # at this position (nonterminal idx, sum of log probability over the constituent)
        inside = [[[] for i in xrange(length + 1)] for j in xrange(length + 1)]

        # a hashmap that stores the total prob of certain constituent
        hash_map = {}

        ## Inside Algorithm
        root_idx = 2

        # Initialization

        # TODO(@Bo) speed up!
        for i in xrange(length):
            child = sen.data[i]
            for parent in xrange(self.nnt):
                if self.lexicon[child][parent]:
                    # new nonterminal found, append to list
                    # calculate each part of the entry
                    log_rule_prob = self.log_prob_left(
                        parent,
                        0,
                        left_context[i]
                    ) + self.log_prob_ut(
                        parent,
                        child,
                        left_context[i]
                    )
                    tpl = (parent, log_rule_prob, -2, child, i)
                    inside[i][i+1].append(tpl)
                    tpl_map = (i, i+1, parent)
                    hash_map[tpl_map] = (len(inside[i][i+1])-1, log_rule_prob)

        # Unary appending, deal with non_term -> non_term ... -> term chain
        for i in xrange(length):
            for child_tpl in inside[i][i+1]:
                child = child_tpl[0]
                previous_log_prob = child_tpl[1]
                for parent in xrange(self.nnt):
                    if self.urules[child][parent] == 1:
                        log_rule_prob = self.log_prob_left(
                            parent,
                            1,
                            left_context[i]
                        ) + self.log_prob_unt(
                            parent,
                            child,
                            left_context[i]
                        )
                        curr_log_prob = previous_log_prob + log_rule_prob
                        tpl_map = (i, i+1, parent)
                        if not tpl_map in hash_map:
                            left_sib = -1
                            tpl = (parent, curr_log_prob, -1, child, i)
                            inside[i][i+1].append(tpl)
                            hash_map[tpl_map] = (len(inside[i][i+1])-1, curr_log_prob)
                        elif curr_log_prob > hash_map[tpl_map][1]:
                            left_sib = -1
                            idx = hash_map[tpl_map][0]
                            tpl = (parent, curr_log_prob, -1, child, i)
                            inside[i][i+1][idx] = tpl
                            hash_map[tpl_map] = (idx, curr_log_prob)

        # viterbi algorithm
        for width in xrange(2, length+1):
            for start in xrange(0, length-width+1):
                end = start + width
                # binary rule
                for mid in xrange(start+1, end):
                    for left_sib_tpl in inside[start][mid]:
                        for child_tpl in inside[mid][end]:
                            left_sib = left_sib_tpl[0]
                            left_sib_log_prob = left_sib_tpl[1]
                            child = child_tpl[0]
                            child_log_prob = child_tpl[1]
                            previous_log_prob = left_sib_log_prob + child_log_prob
                            for parent in xrange(self.nnt):
                                if self.brules[left_sib][child][parent] == 1:
                                    log_rule_prob = self.log_prob_left(
                                        parent,
                                        child,
                                        left_context[start]
                                    ) + self.log_prob_right(
                                        parent,
                                        left_sib,
                                        child,
                                        left_context[mid]
                                    )

                                    curr_log_prob = previous_log_prob + log_rule_prob
                                    tpl_map = (start, end, parent)
                                    if not tpl_map in hash_map:
                                        tpl = (parent, curr_log_prob, left_sib, child, mid)
                                        inside[start][end].append(tpl)
                                        tpl_map = (start, end, parent)
                                        hash_map[tpl_map] = (len(inside[start][end])-1, curr_log_prob)
                                    elif curr_log_prob > hash_map[tpl_map][1]:
                                        idx = hash_map[tpl_map][0]
                                        tpl = (parent, curr_log_prob, left_sib, child, mid)
                                        inside[start][end][idx] = tpl
                                        hash_map[tpl_map] = (idx, curr_log_prob)

                # unary rule
                for child_tpl in inside[start][end]:
                    child = child_tpl[0]
                    previous_log_prob = child_tpl[1]
                    for parent in xrange(self.nnt):
                        if self.urules[child][parent] == 1:
                            log_rule_prob = self.log_prob_left(
                                parent,
                                1,
                                left_context[start]
                            ) + self.log_prob_unt(
                                parent,
                                child,
                                left_context[start]
                            )

                            curr_log_prob = previous_log_prob + log_rule_prob
                            tpl_map = (start, end, parent)
                            left_sib = -1
                            if not tpl_map in hash_map:
                                tpl = (parent, curr_log_prob, -1, child, start)
                                inside[start][end].append(tpl)
                                tpl_map = (start, end, parent)
                                hash_map[tpl_map] = (len(inside[start][end])-1, curr_log_prob)
                            elif curr_log_prob > hash_map[tpl_map][1]:
                                idx = hash_map[tpl_map][0]
                                tpl = (parent, curr_log_prob, -1, child, end)
                                inside[start][end][idx] = tpl
                                hash_map[tpl_map] = (idx, curr_log_prob)

        '''
        # DEBUG
        for x in hash_map:
            print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x]][1].data[0])

        '''
        tpl_map = (0, length, root_idx)
        posterior = 1
        if not tpl_map in hash_map:
            print "No parse at all ! "
            return -1, None, None, -1, -1
        else:
            nll = -inside[0][length][ hash_map[tpl_map][0] ][1]
            return nll, inside, hash_map, length, root_idx


    def unsupervised(self, seq_term):
        emb_inp = self.encoder_t(seq_term)
        output, hidden = self.coef_lstm * self.LSTM(emb_inp, self.h0)

        nll = Variable(torch.FloatTensor([0]))
        for mini_batch in range(len(seq_term)):
            sen = seq_term[mini_batch]
            left_context = output[mini_batch]

            length = len(sen)
            # every entry is a list of tuples, with each tuple indicate a potential nonterminal 
            # at this position (nonterminal idx, sum of log probability over the constituent)
            inside = [[[] for i in xrange(length + 1)] for j in xrange(length + 1)]

            # a hashmap that stores the total prob of certain constituent
            hash_map = {}

            ## Inside Algorithm
            root_idx = 2
            print "Starting the inside algorithm ... "

            # Initialization

            # TODO(@Bo) speed up!
            for i in xrange(length):
                child = sen.data[i]
                for parent in xrange(self.nnt):
                    if self.lexicon[child][parent]:
                        # new nonterminal found, append to list
                        # calculate each part of the entry
                        log_rule_prob = self.log_prob_left(
                            parent,
                            0,
                            left_context[i]
                        ) + self.log_prob_ut(
                            parent,
                            child,
                            left_context[i]
                        )
                        tpl = (parent, log_rule_prob)
                        inside[i][i+1].append(tpl)
                        tpl_map = (i, i+1, parent)
                        hash_map[tpl_map] = len(inside[i][i+1])-1

            # Unary appending, deal with non_term -> non_term ... -> term chain
            for i in xrange(length):
                for child_tpl in inside[i][i+1]:
                    child = child_tpl[0]
                    previous_log_prob = child_tpl[1]
                    for parent in xrange(self.nnt):
                        if self.urules[child][parent] == 1:
                            log_rule_prob = self.log_prob_left(
                                parent,
                                1,
                                left_context[i]
                            ) + self.log_prob_unt(
                                parent,
                                child,
                                left_context[i]
                            )
                            curr_log_prob = previous_log_prob + log_rule_prob
                            tpl_map = (i, i+1, parent)
                            if not tpl_map in hash_map:
                                left_sib = -1
                                tpl = (parent, curr_log_prob)
                                inside[i][i+1].append(tpl)
                                hash_map[tpl_map] = len(inside[i][i+1])-1
                            else:
                                left_sib = -1
                                idx = hash_map[tpl_map]
                                old_log_prob = inside[i][i+1][idx][1]
                                tpl = (parent, self.log_sum_exp(curr_log_prob, old_log_prob))
                                inside[i][i+1][idx] = tpl

            # viterbi algorithm
            for width in xrange(2, length+1):
                for start in xrange(0, length-width+1):
                    end = start + width
                    # binary rule
                    for mid in xrange(start+1, end):
                        for left_sib_tpl in inside[start][mid]:
                            for child_tpl in inside[mid][end]:
                                left_sib = left_sib_tpl[0]
                                left_sib_log_prob = left_sib_tpl[1]
                                child = child_tpl[0]
                                child_log_prob = child_tpl[1]
                                previous_log_prob = left_sib_log_prob + child_log_prob
                                for parent in xrange(self.nnt):
                                    if self.brules[left_sib][child][parent] == 1:
                                        log_rule_prob = self.log_prob_left(
                                            parent,
                                            child,
                                            left_context[start]
                                        ) + self.log_prob_right(
                                            parent,
                                            left_sib,
                                            child,
                                            left_context[mid]
                                        )

                                        curr_log_prob = previous_log_prob + log_rule_prob
                                        tpl_map = (start, end, parent)
                                        if not tpl_map in hash_map:
                                            tpl = (parent, curr_log_prob)
                                            inside[start][end].append(tpl)
                                            tpl_map = (start, end, parent)
                                            hash_map[tpl_map] = len(inside[start][end])-1
                                        else:
                                            idx = hash_map[tpl_map]
                                            old_log_prob = inside[start][end][idx][1]
                                            tpl = (parent, self.log_sum_exp(curr_log_prob, old_log_prob))
                                            inside[start][end][idx] = tpl

                    # unary rule
                    for child_tpl in inside[start][end]:
                        child = child_tpl[0]
                        previous_log_prob = child_tpl[1]
                        for parent in xrange(self.nnt):
                            if self.urules[child][parent] == 1:
                                log_rule_prob = self.log_prob_left(
                                    parent,
                                    1,
                                    left_context[start]
                                ) + self.log_prob_unt(
                                    parent,
                                    child,
                                    left_context[start]
                                )

                                curr_log_prob = previous_log_prob + log_rule_prob
                                tpl_map = (start, end, parent)
                                left_sib = -1
                                if not tpl_map in hash_map:
                                    tpl = (parent, curr_log_prob)
                                    inside[start][end].append(tpl)
                                    tpl_map = (start, end, parent)
                                    hash_map[tpl_map] = len(inside[start][end])-1
                                else:
                                    idx = hash_map[tpl_map]
                                    old_log_prob = inside[start][end][idx][1]
                                    tpl = (parent, self.log_sum_exp(curr_log_prob, old_log_prob))
                                    inside[start][end][idx] = tpl

            print "Finish inside algorithm ... "
            '''
            # DEBUG
            for x in hash_map:
                print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x]][1].data[0])

            '''
            tpl_map = (0, length, root_idx)
            posterior = 1
            if not tpl_map in hash_map:
                print "No parse at all ! "
            else:
                nll -= inside[0][length][hash_map[tpl_map]][1]

        return nll + self.l2()

    def log_prob_ut(self, parent, child, history):
        logsoftmax = nn.LogSoftmax()
        condition = torch.cat((Variable(self.encoder_nt[parent].view(1, -1)), history.view(1, -1)), 1)
        res = self.encoder_t.weight.mm(self.ut(condition).t()).view(1, -1)
        res = logsoftmax(res).view(-1)
        return res[child]

    def log_prob_unt(self, parent, child, history):
        logsoftmax = nn.LogSoftmax()
        condition = torch.cat((Variable(self.encoder_nt[parent].view(1, -1)), history.view(1, -1)), 1)
        res = logsoftmax(self.unt(condition)).view(-1)
        return res[child]

    def log_prob_left(self, parent, child, history):
        logsoftmax = nn.LogSoftmax()
        condition = torch.cat((Variable(self.encoder_nt[parent].view(1, -1)), history.view(1, -1)), 1)
        res = logsoftmax(self.p2l(condition)).view(-1)
        return res[child]

    def log_prob_right(self, parent, left_sib, child, history):
        logsoftmax = nn.LogSoftmax()
        condition = torch.cat(
                (
                    Variable(self.encoder_nt[parent].view(1,-1)),
                    Variable(self.encoder_nt[left_sib].view(1,-1)),
                    history.view(1, -1)
                ), 
            1)
        res = logsoftmax(self.pl2r(condition)).view(-1)
        return res[child]

    def log_sum_exp(self, a, b):
        m = a if a > b else b
        return m + torch.log(torch.exp(a-m) + torch.exp(b-m))

    def supervised(self, seq_term, seq_preterm, 
        p2l, p2l_target, 
        pl2r, pl2r_target, 
        unt, unt_target):


        #t0 = time.time()

        emb_inp = self.encoder_t(seq_term)
        output, hidden = self.coef_lstm * self.LSTM(emb_inp, self.h0) 

        out = output.clone()
        nbatch, height, length, depth = p2l.size()

        output = output.contiguous().view(nbatch, 1, length, -1)
        output = output.repeat(1, height, 1, 1)

        #t1 = time.time()

        p2l = torch.cat((p2l, output), 3)
        pl2r = torch.cat((pl2r, output), 3)
        unt = torch.cat((unt, output), 3)

        logsoftmax = nn.LogSoftmax()
        preterm = torch.cat((seq_preterm, out), 2)
        a, b, c = preterm.size()

        preterm = self.ut(preterm.view(-1, c))
        preterm = preterm.mm(self.encoder_t.weight.t())
        term = seq_term.clone()
        # x.gather(1, b.unsqueeze(1))
        preterm_target = term.view(-1, 1).repeat(1, self.nt)
        nll_pret = -torch.sum(
                torch.gather(logsoftmax(preterm), 1, preterm_target)
            ) / self.nt

        #t2 = time.time()

        a, b, c, d = p2l.size()
        p2l = self.p2l(p2l.view(-1, d))
        p2l_target = p2l_target.view(-1, 1).repeat(1, self.nnt)
        mask = p2l_target >=0
        p2l = p2l.masked_select(mask).view(-1, self.nnt)
        p2l_target = p2l_target.masked_select(mask).view(-1, self.nnt)
        nll_p2l = -torch.sum(
                torch.gather(logsoftmax(p2l), 1, p2l_target)
            ) / self.nnt

        #t3 = time.time()

        a, b, c, d = pl2r.size()
        pl2r = self.pl2r(pl2r.view(-1, d))
        pl2r_target = pl2r_target.view(-1, 1).repeat(1, self.nnt)
        mask = (pl2r_target >= 0)
        pl2r = pl2r.masked_select(mask).view(-1, self.nnt)
        pl2r_target = pl2r_target.masked_select(mask).view(-1, self.nnt)
        nll_pl2r = -torch.sum(
                torch.gather(logsoftmax(pl2r), 1, pl2r_target)
            ) / self.nnt

        #t4 = time.time()

        a, b, c, d = unt.size()
        unt = self.unt(unt.view(-1, d))
        unt_target = unt_target.view(-1, 1).repeat(1, self.nnt)
        mask = (unt_target >= 0)
        unt = unt.masked_select(mask).view(-1, self.nnt)
        unt_target = unt_target.masked_select(mask).view(-1, self.nnt)
        nll_unt = -torch.sum(
                torch.gather(logsoftmax(unt), 1, unt_target)
            ) / self.nnt

        #print "needs %.4f, %.4f, %.4f, %.4f secs" % (round(t1- t0, 0), round(t2- t1, 0), round(t3- t2, 0), round(t4- t3, 0))
        nll = nll_pret + nll_p2l + nll_pl2r + nll_unt

        return nll + self.l2()

    def l2(self):
        l2 = Variable(torch.FloatTensor([0]))
        for param in self.parameters():
            if param.size(0) == 400000:
                l2 += torch.sum(torch.pow(param - Variable(self.term_emb), 2))
            else:
                l2 += torch.sum(torch.pow(param, 2))
        return self.coef_l2 * l2

