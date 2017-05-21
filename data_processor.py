import constants
import nltk
import numpy as np
import os
import time
import torch

from numpy import array
from ptb import ptb

class Processor(object):  

    def __init__(self, train, read_data, verbose):
        self.train_data = train         # the train file
        self.read_data = read_data      # whether read new data
        self.verbose = verbose          # verbose mode or not

        ## Terminals
        self.dt = 100                   # dimension of terminal, as in word2vec
        self.nt = -1                    # number of terminals
        self.w2idx = {}                 # (string -> int)
        self.idx2w = {}                 # (int -> string)

        ## Nonterminals
        self.dnt = -1                   # dimension of nonterminal feature
        self.nnt = -1                   # number of nonterminals
        self.nt2idx = {}                # (string -> int)
        self.idx2nt = []                # (int -> string)


    def read_word2vec(self):
        '''
        This method reads in word embeddings from the Word2vec file 
        and store them into a hash map.        
        '''
        # Get the filename
        w2v_file = constants.W2V_FILE

        self.term_emb = torch.FloatTensor(
            int(constants.MAX_VOCAB_SIZE), self.dt)

        # Deal with OOV (TODO@Bo: change this to be more signature specific)
        self.term_emb[0] = torch.ones(self.dt)  # for OOV
        self.term_emb[1] = torch.zeros(self.dt) # for BOS
        self.w2idx['OOV'] = 0
        self.w2idx['BOS'] = 1
        self.idx2w[0] = 'OOV'
        self.idx2w[1] = 'BOS'

        if os.path.exists(w2v_file):
            begin_time = time.time()
            with open(w2v_file, 'r') as w2v:
                wordIdx = 2
                for line in w2v:
                    embeddingStr = line.split()
                    word = embeddingStr.pop(0)
                    embedding = torch.FloatTensor([float(v_i) for v_i in embeddingStr])
                    self.term_emb[wordIdx] = embedding
                    self.w2idx[word] = wordIdx
                    self.idx2w[wordIdx] = word
                    wordIdx += 1
            # record the size of our vocabulary
            self.nt = wordIdx
            self.term_emb = self.term_emb.narrow(0, 0, self.nt)
            end_time = time.time()
            if self.verbose:
                print "-- Reading word2vec takes %.4f, secs" % round(end_time - begin_time, 5)
        else:
            # the file does not exist
            if self.verbose:
                print "No embeddings of the given dimension," \
                    " the filename is %s" % w2v_file
        return

    ## this method reads and creates the nonterminal embeddings
    def create_nt_emb(self):

        nt_file = constants.NT_EMB_FILE

        if os.path.exists(nt_file):
            begin_time = time.time()
            with open(nt_file, 'r') as nt:
                nt.next() # skip the comment
                self.dnt = self.nnt = int(nt.next().split('\t', 1)[0]) + 2

                self.nonterm_emb = torch.eye(self.nnt, self.dnt)
                if self.verbose:
                    print "The number of nonterminals" \
                    "(include symbols U_TM and U_NTM) is %d" % self.nnt

                # Set up the special symbol UNARY for 
                # unary terminal and nonterminal rules:
                # 0 U_TM
                # 1 U_NTM
                # 2 ROOT
                # ...

                self.nt2idx['U_TM'] = 0
                self.nt2idx['U_NTM'] = 1
                self.idx2nt.append('U_TM')
                self.idx2nt.append('U_NTM')
                self.new_nt_num = 2

                idx = self.new_nt_num
                for n in nt: 
                    nonterminal = n.strip()
                    self.nt2idx[nonterminal] = idx
                    self.idx2nt.append(nonterminal)
                    idx += 1
            end_time = time.time()
            if self.verbose:
                print "-- Reading nonterminals takes %.4f, secs" \
                    % round(end_time - begin_time, 5)
        else:
             # the file does not exist
             if self.verbose:
                print "No such nonterminal embedding file," \
                    " the filename is %s" % nt_file         
        return

    def make_trainset(self):
        examples = ptb("train", minlength=3, maxlength=30, n=10)
        train = list(examples)

        train_file = open(constants.TRAIN, 'w')
        for (sentence, gold_tree) in train:
            train_file.write(sentence + "\n")
            train_file.write(self.convert_tree_to_encoded_list(gold_tree) + "\n")
        # Debug: print self.convert_tree_to_encoded_list(nltk.Tree.fromstring("(ROOT (S (@S (NP I) (VP (VBP live)))(. .)))"))
        
        train_file.close()
                
    def convert_tree_to_encoded_list(self, tree):       
        self.encoded_list = ""
        self.traverseTree(tree, 0, 0)
        return self.encoded_list
    
    def traverseTree(self, tree, depth, width):
        p = tree.label()
        encoded_prefix = " " + str(width) + " " + str(self.nt2idx[p])
        encoded_suffix = ""
        updated_width = width

        if tree.height() == 2:  # is leaf
            type_symb = "t"
            child = tree.leaves()[0]
        else:
            ''' Logic:
            1. give width to left child
            2. get updated_width from left child
            3. give updated_width+1 to left child
            4. get updated_width from right child
            '''
            child_order = -1    # child_order == 0  =>  left child
                                # child_order == 1  =>  right child
            for subtree in tree:
                child_order += 1
                if child_order == 0:
                    child = self.nt2idx[subtree.label()]
                encoded_suffix += " " + str(self.nt2idx[subtree.label()]) # 1st time it will encode left child
                                                                          # 2nd time it will encode right child
                if type(subtree) == nltk.tree.Tree:
                    updated_width = self.traverseTree(subtree, depth+1, updated_width+child_order)
                    if child_order == 1:
                        self.encoded_list += encoded_prefix + encoded_suffix

            if child_order == 0:  # unary rule
                type_symb = "u"
            else:  # binary rule
                type_symb = "l"
        encoded_suffix = " " + type_symb + " " + str(child)
        self.encoded_list += encoded_prefix + encoded_suffix
        return updated_width
    
    ##  read corpus
    def read_corpus(self):

        # lexicon = (nt, nnt)
        # unary_dict = (nnt, nnt)
        # binary_dict = (nnt, nnt, nnt)

        lexicon = {}
        unary = {}
        binary = {}

        if os.path.exists(self.train_data):
            with open(self.train_data, 'r') as data:
                self.lines = data.readlines()
                nsen = len(self.lines) / 2
                assert (len(self.lines) % 2 == 0)
                if self.verbose:
                    print "There are %d sentences in data" % nsen

                for i in xrange(nsen):           
                    # look at the parse
                    parse = self.lines[2*i+1].split()
                    for j in xrange(len(parse)/4):
                        if self.is_digit(parse[4*j+2]):
                            # binary rule
                            parent = int(parse[4*j+1]) + self.new_nt_num
                            left = int(parse[4*j+2]) + self.new_nt_num
                            right = int(parse[4*j+3]) + self.new_nt_num
                            tpl = (left, right)
                            if not tpl in binary:
                                binary[tpl] = set()
                            binary[tpl].add(parent)
                        elif parse[4*j+2] == 'u':
                            # unary nonterminal rule
                            parent = int(parse[4*j+1]) + self.new_nt_num
                            child = int(parse[4*j+3]) + self.new_nt_num
                            if not child in unary:
                                unary[child] = set()
                            unary[child].add(parent)
                        elif parse[4*j+2] == 't':
                            # unary terminal rule
                            parent = int(parse[4*j+1]) + self.new_nt_num
                            word = parse[4*j+3].lower()
                            child = self.w2idx[word] \
                                if word in self.w2idx else 0

                            if not child in lexicon:
                                lexicon[child] = set()
                            lexicon[child].add(parent)     
        else:
            print "No such corpus with filename: %s" % self.train_data

        self.lexicon = {}
        self.unary = {}
        self.binary = {}

        for s in lexicon:
            self.lexicon[s] = list(lexicon[s])

        for u in unary:
            self.unary[u] = list(unary[u])

        for b in binary:
            self.binary[b] = list(binary[b])

        return

    def next(self, idx, bzs=None):
        start = time.time()

        if bzs == None:
            ## unsupervised
            if 2*idx < len(self.lines):
                self.sen = self.get_idx(self.lines[2*idx])
                return idx+1
            else:
                return -1
        else:
            ## supervised

            # bzs is batch size, the number of sentences we process each time

            # c stands for cut_off value here
            c_pl2r = c_ut = constants.MAX_SEN_LENGTH
            c_unt = constants.C_UNT
            c_p2l = constants.C_P2L

            self.sens = torch.LongTensor(bzs, c_ut).fill_(0)

            self.p2l = torch.zeros(bzs, c_p2l, self.dnt)
            self.pl2r = torch.zeros(bzs, c_pl2r, self.dnt*2)
            self.ut = torch.zeros(bzs, c_ut, self.dnt)
            self.unt = torch.zeros(bzs, c_unt, self.dnt)

            # target list, for softmax select
            self.p2l_t = torch.LongTensor(bzs, c_p2l).fill_(-1)
            self.pl2r_t = torch.LongTensor(bzs, c_pl2r).fill_(-1)
            self.ut_t = torch.LongTensor(bzs, c_ut).fill_(-1)
            self.unt_t = torch.LongTensor(bzs, c_unt).fill_(-1)

            # hidden index list
            self.p2l_hi = torch.LongTensor(bzs, c_p2l).fill_(0)
            self.pl2r_hi = torch.LongTensor(bzs, c_pl2r).fill_(0)
            self.ut_hi = torch.LongTensor(bzs, c_ut).fill_(0)
            self.unt_hi = torch.LongTensor(bzs, c_unt).fill_(0)

            senIdx = idx
            senNum = 0

            wrong = False
            length = len(self.lines)
            while senNum < bzs: # gather bzs number of sentences as input
                if 2 * senIdx >= length:
                    wrong = True
                    break

                sen = [w.lower() for w in self.lines[2*senIdx].split()]
                sen = ['BOS'] + sen

                # ignore sentences that are too long
                if len(sen) > constants.MAX_SEN_LENGTH:
                    senIdx += 1
                    continue
                # find a valid sentence
                senNum += 1

                # deal with the rest of inputs
                rest = self.lines[2*senIdx+1].split()

                # index of each
                i_p2l = 0
                i_pl2r = 0
                i_ut = 0
                i_unt = 0

                for j in xrange(len(rest)/4):
                    if rest[4*j+2] == 'u' and i_unt < c_unt and i_p2l < c_p2l:
                        # unary nonterminal rule
                        self.unt[senNum-1][i_unt] = self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num]
                        self.unt_t[senNum-1][i_unt] = int(rest[4*j+3]) + self.new_nt_num
                        self.unt_hi[senNum-1][i_unt] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_unt += 1

                        self.p2l[senNum-1][i_p2l] = self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num]
                        self.p2l_t[senNum-1][i_p2l] = 1
                        self.p2l_hi[senNum-1][i_p2l] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_p2l += 1
                    elif rest[4*j+2] == 'l' and i_p2l < c_p2l:
                        # left part of binary rule
                        self.p2l[senNum-1][i_p2l] = self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num]
                        self.p2l_t[senNum-1][i_p2l] = int(rest[4*j+3]) + self.new_nt_num
                        self.p2l_hi[senNum-1][i_p2l] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_p2l += 1
                    elif rest[4*j+2] == 't' and i_ut < c_ut and i_p2l < c_p2l:
                        # terminal rule
                        self.ut[senNum-1][i_ut] = self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num]
                        try:
                            self.ut_t[senNum-1][i_ut] = self.w2idx[rest[4*j+3].lower()]
                            self.sens[senNum-1][i_ut] = self.w2idx[rest[4*j+3].lower()]
                        except KeyError:
                            self.ut_t[senNum-1][i_ut] = 0
                            self.sens[senNum-1][i_ut] = 0
                        self.ut_hi[senNum-1][i_ut] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_ut += 1

                        self.p2l[senNum-1][i_p2l] = self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num]
                        self.p2l_t[senNum-1][i_p2l] = 0
                        self.p2l_hi[senNum-1][i_p2l] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_p2l += 1
                    elif self.is_digit(rest[4*j+2]) and i_pl2r < c_pl2r:
                        # right part of binary rule
                        temp = torch.cat((
                                self.nonterm_emb[int(rest[4*j+1]) + self.new_nt_num].view(-1),
                                self.nonterm_emb[int(rest[4*j+2]) + self.new_nt_num].view(-1)
                            ), 0)

                        self.pl2r[senNum-1][i_pl2r] = temp
                        self.pl2r_t[senNum-1][i_pl2r] = int(rest[4*j+3]) + self.new_nt_num
                        self.pl2r_hi[senNum-1][i_pl2r] = int(rest[4*j]) + (senNum-1) * c_ut
                        i_pl2r += 1
                senIdx += 1
            end = time.time()
            if self.verbose:
                print " - Extracting input takes %.4f" % round(end - start, 5)

            return -1 if wrong else senIdx

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

        #print len(unt_p)

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

        #print len(p2l_p)
        #print len(pl2r_p)

    def read_and_process(self):
        if self.read_data:
            if os.path.exists(constants.CORPUS_INFO_FILE):
                os.remove(constants.CORPUS_INFO_FILE)
            if self.verbose:
                print "Reading and processing data ... "
            start = time.time()

            self.read_word2vec()
            self.create_nt_emb()
            self.read_corpus()
            self.create_precomputed_matrix()

            end = time.time()

            self.print_rules()

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
            self.new_nt_num = 2
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
