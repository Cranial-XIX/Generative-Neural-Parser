import itertools
import math
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
        self.term_emb = inputs['term_emb']      # embeddings of terminals
        self.nt = inputs['nt']                  # number of terminals
        self.dt = inputs['dt']                  # dimension of terminals

        # nonterminals
        self.nonterm_emb = inputs['nt_emb']     # embeddings of nonterminals
        self.nnt = inputs['nnt']                # number of nonterminals
        self.dnt = inputs['dnt']                # dimension of nonterminals

        # model
        self.cuda_flag = cuda_flag

        self.coef_lstm = inputs['coef_lstm']    # coefficient of LSTM
        self.bsz = inputs['bsz']                # the batch size
        self.dhid = inputs['dhid']              # dimension of hidden layer
        self.nlayers = inputs['nlayers']        # number of layers in neural net
        initrange = inputs['initrange']         # range for uniform initialization
        self.urules = inputs['urules']          # dictionary of unary rules
        self.brules = inputs['brules']          # dictionary of binary rules
        self.lexicon = inputs['lexicon']        # dictionary of lexicon

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

        if self.cuda_flag:
            nll = Variable(torch.FloatTensor([0])).cuda()
        else:
            nll = Variable(torch.FloatTensor([0]))

        sen = sen.view(-1)
        left_context = self.coef_lstm * output[0]

        length = len(sen) - 1
        # every entry is a list of tuples, with each tuple indicate a potential nonterminal 
        # at this position (nonterminal idx, sum of log probability over the constituent)
        inside = [[[] for i in xrange(length + 1)] for j in xrange(length + 1)]

        # a hashmap that stores the total prob of certain constituent
        hash_map = {}

        ## Inside Algorithm
        root_idx = 2

        # Initialization

        # TODO(@Bo) speed up!
        tt0 = time.time()
        for i in xrange(length):
            child = sen.data[i+1]
            for parent in self.lexicon[child]:
                # new nonterminal found, append to list
                # calculate each part of the entry
                log_rule_prob = self.log_prob_left(
                        parent, 0, left_context[i]
                    ) + self.log_prob_ut(
                        parent, child, left_context[i]
                    )
                tpl = (parent, log_rule_prob, -2, child, i)
                inside[i][i+1].append(tpl)
                tpl_map = (i, i+1, parent)
                hash_map[tpl_map] = (len(inside[i][i+1])-1, log_rule_prob)
        tt1 = time.time()
        if self.verbose == 'yes':
            print "LEXICON ", tt1-tt0, "---------------------------"
         
        tt0 = time.time()
        # Unary appending, deal with non_term -> non_term ... -> term chain
        for i in xrange(length):
            for child_tpl in inside[i][i+1]:
                child = child_tpl[0]
                previous_log_prob = child_tpl[1]
                if child in self.urules:
                    for parent in self.urules[child]:
                        log_rule_prob = self.log_prob_left(
                                parent, 1, left_context[i]
                            ) + self.log_prob_unt(
                                parent, child, left_context[i]
                            )
                        curr_log_prob = previous_log_prob + log_rule_prob
                        tpl_map = (i, i+1, parent)
                        if not tpl_map in hash_map:
                            left_sib = -1
                            tpl = (parent, curr_log_prob, -1, child, i)
                            inside[i][i+1].append(tpl)
                            hash_map[tpl_map] = (len(inside[i][i+1])-1, curr_log_prob)
        tt1 = time.time()
        if self.verbose == 'yes':
            print "Unary appending ", tt1-tt0, "---------------------------"
            
        # viterbi algorithm
        tt0 = time.time()
        for width in xrange(2, length+1):
            for start in xrange(0, length-width+1):
                end = start + width
                # binary rule
                t00 = time.time()
                for mid in xrange(start+1, end):
                    for left_sib_tpl in inside[start][mid]:
                        for child_tpl in inside[mid][end]:
                            left_sib = left_sib_tpl[0]
                            left_sib_log_prob = left_sib_tpl[1]
                            child = child_tpl[0]
                            child_log_prob = child_tpl[1]
                            previous_log_prob = left_sib_log_prob + child_log_prob
                            children = (left_sib, child)
                            if children in self.brules:
                                for parent in self.brules[children]:
                                    log_rule_prob = self.log_prob_left(
                                            parent, child, left_context[start]
                                        ) + self.log_prob_right(
                                            parent, left_sib, child, left_context[mid]
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
                t01 = time.time()
                if self.verbose == 'yes':
                    print "Binary rules ", t01-t00, "---------------------------"                                        

                # unary rule
                t00 = time.time()
                for child_tpl in inside[start][end]:
                    child = child_tpl[0]
                    previous_log_prob = child_tpl[1]
                    if child in self.urules:
                        for parent in self.urules[child]:
                                log_rule_prob = self.log_prob_left(
                                        parent, 1, left_context[start]
                                    ) + self.log_prob_unt(
                                        parent, child, left_context[start]
                                    )
                                curr_log_prob = previous_log_prob + log_rule_prob
                                tpl_map = (start, end, parent)
                                left_sib = -1
                                if not tpl_map in hash_map:
                                    tpl = (parent, curr_log_prob, -1, child, start)
                                    inside[start][end].append(tpl)
                                    tpl_map = (start, end, parent)
                                    hash_map[tpl_map] = (len(inside[start][end])-1, curr_log_prob)
                t01 = time.time()
                if self.verbose == 'yes':
                    print "Unary rules ", t01-t00, "---------------------------"
        tt1 = time.time()
        if self.verbose == 'yes':
            print "VITERBI ", tt1-tt0, "---------------------------"
            
        tpl_map = (0, length, root_idx)
        posterior = 1
        if not tpl_map in hash_map:
            # DEBUG
            #for x in hash_map:
            #    print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x][0]][1].data[0])
            return -1, None, None, -1, -1
        else:
            nll = -inside[0][length][ hash_map[tpl_map][0] ][1]
            # DEBUG
            #if self.verbose == 'yes':
            #    for x in hash_map:
            #        print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x][0]][1].data[0])
            return nll, inside, hash_map, length, root_idx
        

    def unsupervised(self, sen):
        emb_inp = self.encoder_t(sen)
        output, hidden = self.LSTM(emb_inp, self.h0)

        left_context = self.coef_lstm * output.squeeze(0)

        sen = sen.view(-1)
        length = len(sen)
        # every entry is a list of tuples, with each tuple indicate
        # a potential nonterminal at this position 
        # (nonterminal idx, sum of log probability over the constituent)
        inside = [[[] for i in xrange(length+1)] for j in xrange(length+1)]

        # a hashmap that stores the total prob of certain constituent
        hash_map = {}

        ## Inside Algorithm

        root_idx = 2

        c = 0
        for i in self.urules:
            c += len(self.urules[i])
        for p in self.brules:
            c += len(self.brules[p])
        if self.verbose == 'yes':
            print "size of grammar : ", c
        # Initialization
        tt0 = time.time()
        for i in xrange(length):
            child = sen.data[i]
            for parent in self.lexicon[child]:
                # new nonterminal found, append to list
                # calculate each part of the entry
                log_rule_prob = self.log_prob_left(
                        parent, 0, left_context[i]
                    ) + self.log_prob_ut(
                        parent, child, left_context[i]
                    )
                tpl = (parent, log_rule_prob)
                inside[i][i+1].append(tpl)
                tpl_map = (i, i+1, parent)
                hash_map[tpl_map] = len(inside[i][i+1])-1
        tt1 = time.time()
        if self.verbose == 'yes':
            print "LEXICON ", tt1-tt0, "---------------------------"

        # Unary appending, deal with non_term -> non_term ... -> term chain
        tt2 = time.time()
        for i in xrange(length):
            for child_tpl in inside[i][i+1]:
                child = child_tpl[0]
                previous_log_prob = child_tpl[1]
                if child in self.urules:
                    for parent in self.urules[child]:
                        log_rule_prob = self.log_prob_left(
                                parent, 1, left_context[i]
                            ) + self.log_prob_unt(
                                parent, child, left_context[i]
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
        tt3 = time.time()
        if self.verbose == 'yes':
            print "UNARY ", tt3-tt2, "---------------------------"
                            
        if self.verbose == 'yes':
            print 'Viterbi algorithm starting'
        # viterbi algorithm
        tt4 = time.time()
        for width in xrange(2, length+1):
            for start in xrange(0, length-width+1):
                if self.verbose == 'yes':
                    print width, " ", start, " binary rules"
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
                            children = (left_sib, child)
                            if children in self.brules:
                                for parent in self.brules[children]:
                                    log_rule_prob = self.log_prob_left(
                                            parent, child, left_context[start]
                                        ) + self.log_prob_right(
                                            parent, left_sib, child, left_context[mid]
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
                if self.verbose == 'yes':
                    print width, " ", start, " unary rules"
                for child_tpl in inside[start][end]:
                    child = child_tpl[0]
                    previous_log_prob = child_tpl[1]
                    if child in self.urules:
                        for parent in self.urules[child]:
                            log_rule_prob = self.log_prob_left(
                                    parent, 1, left_context[start]
                                ) + self.log_prob_unt(
                                    parent, child, left_context[start]
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
        tt5 = time.time()
        if self.verbose == 'yes':
            print "Finish inside algorithm ... ", tt5 - tt4
        
        tpl_map = (0, length, root_idx)
        if not tpl_map in hash_map:
            # DEBUG
            #for x in hash_map:
            #    print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x]][1].data[0])
            if self.cuda_flag:
                return Variable(torch.FloatTensor([-1])).cuda()
            return Variable(torch.FloatTensor([-1]))

        else:
            return -inside[0][length][hash_map[tpl_map]][1]

    def log_prob_ut(self, parent, child, history):
        t0 = time.time()
        logsoftmax = nn.LogSoftmax()
        if self.cuda_flag:
            parent_LongTensor = Variable(torch.LongTensor([parent])).cuda()
        else:
            parent_LongTensor = Variable(torch.LongTensor([parent]))
        condition = torch.cat((
                self.encoder_nt(parent_LongTensor), 
                history.view(1, -1)
            ), 1)
        t1 = time.time()
        #res = (self.word2vec_plus.weight).mm(self.ut(condition).t()).view(1, -1)
        res = (self.word2vec.weight + self.word2vec_plus.weight).mm(self.ut(condition).t()).view(1, -1)
        t2 = time.time()
        res = logsoftmax(res).view(-1)
        t3 = time.time()
        if self.verbose == 'yes':
            print "log_prob_ut ", parent, " ", child, " ", t1-t0, " ", t2-t1, " ", t3-t2
        return res[child]

    def log_prob_unt(self, parent, child, history):
        dragon = time.time()
        logsoftmax = nn.LogSoftmax()
        if self.cuda_flag:
            parent_LongTensor = Variable(torch.LongTensor([parent])).cuda()
        else:
            parent_LongTensor = Variable(torch.LongTensor([parent]))
        condition = torch.cat((
                self.encoder_nt(parent_LongTensor), 
                history.view(1, -1)
            ), 1)
        res = logsoftmax(self.unt(condition)).view(-1)
        titi = time.time()
        if self.verbose == 'yes':
            pass#print "log_prob_unt ", titi-dragon
        return res[child]

    def log_prob_left(self, parent, child, history):
        dragon = time.time()
        logsoftmax = nn.LogSoftmax()
        if self.cuda_flag:
            parent_LongTensor = Variable(torch.LongTensor([parent])).cuda()
        else:
            parent_LongTensor = Variable(torch.LongTensor([parent]))
        condition = torch.cat((
                self.encoder_nt(parent_LongTensor), 
                history.view(1, -1)
            ), 1)
        res = logsoftmax(self.p2l(condition)).view(-1)
        titi = time.time()
        if self.verbose == 'yes':
            pass#print "log_prob_left ", titi-dragon
        return res[child]

    def log_prob_right(self, parent, left_sib, child, history):
        dragon = time.time()
        logsoftmax = nn.LogSoftmax()
        if self.cuda_flag:
            parent_LongTensor = Variable(torch.LongTensor([parent])).cuda()
            left_sib_LongTensor = Variable(torch.LongTensor([left_sib])).cuda()
        else:
            parent_LongTensor = Variable(torch.LongTensor([parent]))
            left_sib_LongTensor = Variable(torch.LongTensor([left_sib]))
        condition = torch.cat(
                (
                    self.encoder_nt(parent_LongTensor), 
                    self.encoder_nt(left_sib_LongTensor), 
                    history.view(1, -1)
                ), 
            1)
        res = logsoftmax(self.pl2r(condition)).view(-1)
        titi = time.time()
        if self.verbose == 'yes':
            pass#print "log_prob_right ", titi-dragon
        return res[child]

    def log_sum_exp(self, a, b):
        m = a if a > b else b
        return m + torch.log(torch.exp(a-m) + torch.exp(b-m))

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

