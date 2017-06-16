import constants
import nltk
import numpy as np
import os
import random
import time
import torch

from numpy import array
from ptb import ptb

class Processor(object):  

    def __init__(self, train_file, make_train, read_data, verbose):
        self.train_file = train_file    # the train file
        self.read_data = read_data      # whether to read new data
        self.verbose = verbose          # verbose mode or not
        self.make_train = make_train    # whether to make train set

        ## Terminals
        self.dt = 100                   # dimension of terminal, as in word2vec
        self.nt = -1                    # number of terminals
        self.w2idx = {}                 # (string -> int)
        self.idx2w = []                 # (int -> string)

        ## Nonterminals
        self.dnt = -1                   # dimension of nonterminal feature
        self.nnt = -1                   # number of nonterminals
        self.nt2idx = {}                # (string -> int)
        self.idx2nt = []                # (int -> string)

        ## Lexicon and grammar
        self.lexicon = {}               # (int -> [int])
        self.unary = {}                 # (int -> [int])
        self.binary = {}                # ((int,int) -> [int])


    def check_file_exists(self, filename, file_type):
        '''
        Check if a file with given name exists.
        @param file_type Type of file
        @param filename  Name of file
        @return          True if the file exists, False otherwise
        '''
        if not os.path.exists(filename):
            if self.verbose:
                print "No such %s file," \
                    " the filename is %s" % (file_type, filename)
            return False
        return True

    def read_w2v_file(self, w2v_file):
        '''
        Reads in word embeddings from the Word2vec file 
        and store them into a hash map.    
        '''
        if not self.check_file_exists(w2v_file, "word2vec"):
            return

        self.term_emb = torch.FloatTensor(
            int(constants.MAX_VOCAB_SIZE), self.dt)

        # Add OOV and BOS symbols
        self.w2idx[constants.OOV] = 0
        self.w2idx[constants.BOS] = 1
        self.idx2w.append(constants.OOV)
        self.idx2w.append(constants.BOS)
        self.term_emb[0] = torch.ones(self.dt)  # for OOV
        self.term_emb[1] = torch.zeros(self.dt) # for BOS

  
        begin_time = time.time()        
        with open(w2v_file, 'r') as w2v_f:
            w_idx = 2
            for line in w2v_f:
                w2v_str = line.split()
                w = w2v_str.pop(0)
                self.w2idx[w] = w_idx
                self.idx2w.append(w)
                w_emb = torch.FloatTensor(map(float, w2v_str))
                self.term_emb[w_idx] = w_emb
                w_idx += 1
        # record the size of our vocabulary
        self.nt = w_idx
        self.term_emb = self.term_emb.narrow(0, 0, self.nt)
        end_time = time.time()

        if self.verbose:
            print "-- Reading word2vec takes %.4f s" % round(end_time - begin_time, 5)
            print "   # words: ", self.nt
        return
 
    def read_nt_file(self, nt_file):
        '''
        Reads and creates the nonterminal embeddings
        '''
        if not self.check_file_exists(nt_file, "nonterminal embeddings"):
             return

        begin_time = time.time()
        with open(nt_file, 'r') as nt_f:
            nt_f.next()  # skip the comment
            self.dnt = self.nnt = int(nt_f.next()) + 2  # +2 for U_TM & U_NTM

            self.nonterm_emb = torch.eye(self.nnt, self.dnt)
            if self.verbose:
                print "# nonterminals " \
                "(include symbols U_TM and U_NTM): %d" % self.nnt

            # Set up the special symbol UNARY for 
            # unary terminal and nonterminal rules:
            # 0 U_TM
            # 1 U_NTM
            # 2 ROOT
            # ...
            self.nt2idx[constants.U_TM] = 0
            self.nt2idx[constants.U_NTM] = 1
            self.idx2nt.append(constants.U_TM)
            self.idx2nt.append(constants.U_NTM)
            
            idx = 2
            for line in nt_f: 
                nt = line.strip()
                self.nt2idx[nt] = idx
                self.idx2nt.append(nt)
                idx += 1
        end_time = time.time()

        if self.verbose:
            print "-- Reading nonterminals takes %.4f s" \
                % round(end_time - begin_time, 5)      
        return

    def read_lex_file(self, lex_file):
        if not self.check_file_exists(lex_file, "lexicon"):
             return

        npt = 0 # number of preterminal rules
        begin_time = time.time()
        oov_set = set()
        with open(lex_file, 'r') as lex_f:
            for line in lex_f:
                npt += 1
                lex = line.strip().split()
                nt_idx = self.nt2idx[lex[0]]
                w = lex[1]
                if w in self.w2idx:
                    w_idx = self.w2idx[w]
                else:  # if word is OOV
                    w_idx = constants.OOV_IDX
                if not w_idx in self.lexicon:
                    self.lexicon[w_idx] = []
                if w_idx == constants.OOV_IDX:
                    oov_set.add(nt_idx)
                else:
                    self.lexicon[w_idx].append(nt_idx)
        self.lexicon[constants.OOV_IDX] = list(oov_set)
        end_time = time.time()

        if self.verbose:
            print "-- Reading lexicon takes %.4f s" \
                % round(end_time - begin_time, 5)
            print "   # preterminal rules: ", npt
        return
    
    def read_gr_file(self, gr_file):
        if not self.check_file_exists(gr_file, "grammar"):
             return

        nu = 0 # number of unary rules
        nb = 0 # number of binary rules
        begin_time = time.time()
        with open(gr_file, 'r') as gr_f:
            for line in gr_f:
                rule = line.strip().split()
                p = self.nt2idx[rule[0][:-2]]  # [:-2] is to remove "_0" from "NP_0" to form "NP"
                l = self.nt2idx[rule[2][:-2]]
                if len(rule) == 5:                     # binary rule
                    nb += 1
                    r = self.nt2idx[rule[3][:-2]]
                    tpl = (l,r)
                    if not tpl in self.binary:
                        self.binary[tpl] = []
                    self.binary[tpl].append(p)
                if len(rule) == 4:                     # unary rule
                    if p != l:                         # Do not allow self-recurring X -> X rules
                        nu += 1
                        if not l in self.unary:
                            self.unary[l] = []
                        self.unary[l].append(p)
        end_time = time.time()

        if self.verbose:
            print "-- Reading grammar takes %.4f s" \
                % round(end_time - begin_time, 5)
            print "   # binary rules: ", nb
            print "   # unary rules: ", nu            
        return
    
    def read_train_data(self):
        if not self.check_file_exists(self.train_file, "training data"):
            return

        with open(self.train_file, 'r') as data:
            self.lines = data.readlines()    

    def make_trainset(self):
        examples = ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH,n=100)
        train_trees = list(examples)

        f = open(self.train_file, 'w')
        begin_time = time.time()
        first = True
        for (sentence, gold_tree) in train_trees:
            if self.containOOV(sentence):
                continue
            if first:
                f.write(sentence)
                first = False
            else:
                f.write("\n" + sentence)
            f.write("\n" + self.convert_tree_to_encoded_list(gold_tree))
        # DEBUG: print self.convert_tree_to_encoded_list(nltk.Tree.fromstring("(ROOT (S (@S (NP I) (VP (VBP live)))(. .)))"))
        end_time = time.time()
        print "-- Making trainset takes %.4f s" \
            % round(end_time - begin_time, 5)
        f.close()
                
    def convert_tree_to_encoded_list(self, tree):       
        self.encoded_list = [""]
        self.wi = -1 # word index
        self.traverseTree(tree)
        return " ".join(self.encoded_list)

    def traverseTree(self, tree):
        p = str(self.nt2idx[tree.label()])
        if tree.height() == 2:  # is leaf
            child = tree.leaves()[0]
            self.wi += 1
            self.encoded_list.append(str(self.wi))
            self.encoded_list.append(p)
            self.encoded_list.append("t")
            self.encoded_list.append(child)
            self.encoded_list.append("_")
            return self.wi, p
        else:
            nchild = 0
            for subtree in tree:
                if nchild == 0:
                    position, child = self.traverseTree(subtree)
                else:
                    mid, right = self.traverseTree(subtree)
                nchild += 1
            self.encoded_list.append(str(position))
            self.encoded_list.append(p)
            if nchild == 1:
                # unary rule
                self.encoded_list.append("u")
                self.encoded_list.append(child)
                self.encoded_list.append("_")
            else:
                # binary rule
                self.encoded_list.append(child)
                self.encoded_list.append(right)
                self.encoded_list.append(str(mid))
            return position, p

    def containOOV(self, sentence):
        sentence = sentence.strip().split()
        for i in xrange(len(sentence)):
            if sentence[i].lower() not in self.w2idx:
                return True
        return False

    def shuffle(self):
        sens = self.lines[::2]
        rules = self.lines[1::2]
        c = list(zip(sens, rules))
        random.shuffle(c)
        sens, rules = zip(*c)
        self.lines[::2] = sens
        self.lines[1::2] = rules

    def next(self, idx, bsz=None):
        '''
        this function extract the next batch of training instances
        and save them for later use
        '''
        if bsz == None:
            ## unsupervised
            if 2*idx < len(self.lines):
                self.sen = self.get_idx(self.lines[2*idx])
                return idx+1
            else:
                return -1
        else:
            ## supervised
            length = len(self.lines)

            # bsz is batch size, the number of sentences we process each time
            # the maximum number of training instances in a batch
            m = constants.MAX_SEN_LENGTH
            cutoff = bsz * (m+5)

            self.sens = torch.LongTensor(bsz, m).fill_(0)

            self.p2l = torch.LongTensor(cutoff*3,)
            self.ut = torch.LongTensor(cutoff,)
            self.unt = torch.LongTensor(cutoff,)
            self.pl2r_p = torch.LongTensor(cutoff,)
            self.pl2r_l = torch.LongTensor(cutoff,)

            # target list, for softmax select
            self.p2l_t = torch.LongTensor(cutoff*3,)
            self.pl2r_t = torch.LongTensor(cutoff,)
            self.ut_t = torch.LongTensor(cutoff,)
            self.unt_t = torch.LongTensor(cutoff,)

            # hidden index list
            self.p2l_i = torch.LongTensor(cutoff*3,)
            self.pl2r_pi = torch.LongTensor(cutoff,)
            self.pl2r_ci = torch.LongTensor(cutoff,)
            self.ut_i = torch.LongTensor(cutoff,)
            self.unt_i = torch.LongTensor(cutoff,)

            num_p2l = num_pl2r = num_ut = num_unt = num_sen = 0

            while num_sen < bsz and 2*(idx+1) <= length:
                # get the encoded sentence, exclude the last word
                # since we only need left context
                self.sens[num_sen] = self.get_idx_maxlength(self.lines[2*idx])

                # deal with the rest of inputs
                rest = self.lines[2*idx+1].split()

                for j in xrange(len(rest)/5):
                    previous = num_sen * m
                    li = previous + int(rest[5*j])      # left index in matrix
                    p = int(rest[5*j+1])                # parent index
                    symbol = rest[5*j+2]                # might be from: {
                                                        # 't'    (unary terminal rule)
                                                        # 'u'    (unary nontemrinal rule)
                                                        # number (the left sibling) }
                    if symbol == 't':
                        # terminal rule found
                        word = rest[5*j+3].lower()
                        c = self.w2idx[word] if word in self.w2idx else 0
                        self.ut[num_ut] = p
                        self.ut_t[num_ut] = c
                        self.ut_i[num_ut] = li
                        l = 0
                        num_ut += 1
                    elif symbol == "u":
                        # unary nonterminal rule found
                        c = int(rest[5*j+3])
                        self.unt[num_unt] = p
                        self.unt_t[num_unt] = c
                        self.unt_i[num_unt] = li
                        l = 1
                        num_unt += 1
                    else:
                        # binary rule
                        c = int(rest[5*j+3])
                        l = int(symbol)
                        self.pl2r_p[num_pl2r] = p
                        self.pl2r_l[num_pl2r] = l
                        self.pl2r_t[num_pl2r] = c
                        self.pl2r_pi[num_pl2r] = li
                        self.pl2r_ci[num_pl2r] = previous + int(rest[5*j+4])
                        num_pl2r += 1

                    self.p2l[num_p2l] = p
                    self.p2l_t[num_p2l] = l
                    self.p2l_i[num_p2l] = li
                    num_p2l += 1

                num_sen += 1
                idx += 1

            self.p2l = self.p2l[:num_p2l]
            self.ut = self.ut[:num_ut]
            self.unt = self.unt[:num_unt]
            self.pl2r_p = self.pl2r_p[:num_pl2r]
            self.pl2r_l = self.pl2r_l[:num_pl2r]

            # target list, for softmax select
            self.p2l_t = self.p2l_t[:num_p2l]
            self.ut_t = self.ut_t[:num_ut]
            self.unt_t = self.unt_t[:num_unt]
            self.pl2r_t = self.pl2r_t[:num_pl2r]

            # hidden index list
            self.p2l_i = self.p2l_i[:num_p2l]
            self.ut_i = self.ut_i[:num_ut]
            self.unt_i = self.unt_i[:num_unt]
            self.pl2r_pi = self.pl2r_pi[:num_pl2r]
            self.pl2r_ci = self.pl2r_ci[:num_pl2r]

            if 2*(idx+1) > length:
                return -1
            else:
                return idx

    def is_digit(self, n):
        try:
            int(n)
            return True
        except ValueError:
            return False

    def print_rules(self):
        if self.verbose:
            nl = 0
            for word in self.lexicon:
                for NT in self.lexicon[word]:
                    nl += 1
                    print "%s ---> %s" \
                        % (self.idx2nt[NT], self.idx2w[word])                   
            print "Lexicon size : ", nl
            print "-" * 80

            nu = 0
            for child in self.unary:
                for parent in self.unary[child]:
                    nu += 1
                    print "%s ---> %s" \
                        % (self.idx2nt[parent], self.idx2nt[child])
            print "Unary nonterminal rules in total : ", nu
            print "-" * 80

            nb = 0
            for key in self.binary:
                for parent in self.binary[key]:
                    nb += 1
                    print "%s ---> %s %s" % (
                            self.idx2nt[parent], 
                            self.idx2nt[key[0]], 
                            self.idx2nt[key[1]]
                        )
            print "Binary rules in total : ", nb

    def create_precomputed_matrix(self):
        self.unt_pre = torch.FloatTensor(self.nnt, self.nnt).zero_()
        self.p2l_pre = torch.FloatTensor(self.nnt, self.nnt).zero_()
        self.pl2r_pre = torch.FloatTensor(self.nnt, self.nnt, 2*self.nnt).zero_()

        unt_p = set()
        p2l_p = set()
        pl2r_p = set()

        for child in self.unary:
            for parent in self.unary[child]:
                if not parent in unt_p:
                    self.unt_pre[parent] = self.nonterm_emb[parent]
                    unt_p.add(parent)

        self.p2l_pre[0] = self.nonterm_emb[0]
        self.p2l_pre[1] = self.nonterm_emb[1]

        for key in self.binary:
            for parent in self.binary[key]:
                if not parent in p2l_p:
                    p2l_p.add(parent)
                    self.p2l_pre[parent] = self.nonterm_emb[parent]
                if not (parent, key[0]) in pl2r_p:
                    pl2r_p.add((parent, key[0]))
                    self.pl2r_pre[parent][key[0]] = \
                        torch.cat((self.nonterm_emb[parent], self.nonterm_emb[key[0]]), 0).view(1, -1)

    def process_data(self):
        if self.read_data:
            if os.path.exists(constants.CORPUS_INFO_FILE):
                os.remove(constants.CORPUS_INFO_FILE)
            if self.verbose:
                print "Reading and processing data ... "
            start = time.time()

            self.read_w2v_file(constants.W2V_FILE)
            self.read_nt_file(constants.NT_EMB_FILE)
            self.read_lex_file(constants.LEX_FILE)
            self.read_gr_file(constants.GR_FILE)

            if self.make_train:
                self.make_trainset()
            self.read_train_data()
        
            self.create_precomputed_matrix()

            end = time.time()

            # DEBUG: self.print_rules()

            # save those for future use
            torch.save({
                    'term_emb': self.term_emb,
                    'nonterm_emb': self.nonterm_emb,
                    'nt': self.nt,
                    'dt': self.dt,
                    'nnt': self.nnt,
                    'dnt': self.dnt,
                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,
                    'lexicon': self.lexicon,
                    'unary': self.unary,
                    'binary': self.binary,
                    'lines': self.lines,
                    'unt_pre': self.unt_pre,
                    'p2l_pre': self.p2l_pre,
                    'pl2r_pre': self.pl2r_pre
                }, constants.CORPUS_INFO_FILE)
        else:
            # read existing data, so we don't need to process again
            if self.verbose:
                print "Reading existing data ... "
            start = time.time()      
            if not os.path.exists(constants.CORPUS_INFO_FILE):
                print "Error, no corpus info file found"
                return
            d = torch.load(constants.CORPUS_INFO_FILE)
            self.term_emb = d['term_emb']
            self.nonterm_emb = d['nonterm_emb']
            self.nt = d['nt']
            self.dt = d['dt']
            self.nnt = d['nnt']
            self.dnt = d['dnt']
            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']
            self.lines = d['lines']
            self.lexicon = d['lexicon']
            self.unary = d['unary']
            self.binary = d['binary']
            self.unt_pre = d['unt_pre']
            self.p2l_pre = d['p2l_pre']
            self.pl2r_pre = d['pl2r_pre']
            #self.print_rules()
            end = time.time()
        if self.verbose:
            print "Reading data takes %.4f secs" % round(end - start, 5)
 
    def get_sen(self, indices):
        return " ".join([self.idx2w[i] for i in indices])

    def get_idx(self, sen):
        sen_w = ['BOS'] + [w.lower() for w in sen.split()]
        sen_i = torch.LongTensor(1, len(sen_w))
        for i in xrange(len(sen_w)):
            try:
                sen_i[0][i] = self.w2idx[sen_w[i]]
            except KeyError:
                sen_i[0][i] = 0
        return sen_i

    def get_idx_maxlength(self, sen):
        sen_w = ['BOS'] + [w.lower() for w in sen.split()]
        sen_i = torch.LongTensor(1, constants.MAX_SEN_LENGTH).fill_(0)
        for i in xrange(len(sen_w)-1):
            try:
                sen_i[0][i] = self.w2idx[sen_w[i]]
            except KeyError:
                sen_i[0][i] = 0
        return sen_i

