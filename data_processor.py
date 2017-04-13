import numpy as np
from numpy import array
import os
import time
import torch

import constants

class Processor(object):  

    def __init__(self, cmd_inp):
        self.file_data = cmd_inp['data']
        self.read_data = cmd_inp['rd']

        ## Terminals
        self.dt = cmd_inp['dt']         # dimension of terminal, as in word2vec
        self.nt = -1                    # number of terminals
        self.word2Idx = {}              # (string -> int)
        self.idx2Word = {}              # (int -> string)

        ## Nonterminals
        self.dnt = -1                   # dimension of nonterminal feature
        self.nnt = -1                   # number of nonterminals
        self.nonterm2Idx = {}           # (string -> int)
        self.idx2Nonterm = {}           # (int -> string)


    ## This method reads in word embeddings from the Word2vec file 
    ## and store them into a hash map.
    def read_word2vec(self):        
        # Get the filename
        w2v_file = "%s%d%s"  % (constants.WORD_EMBEDDING_FILE_PREFIX, 
            self.dt, constants.WORD_EMBEDDING_FILE_SUFFIX)

        self.term_emb = torch.FloatTensor(int(constants.MAX_VOCAB_SIZE), self.dt)

        # Deal with OOV (TODO@Bo: change this to be more signature specific)
        self.term_emb[0] = torch.ones(self.dt)  # for OOV
        self.term_emb[1] = torch.zeros(self.dt) # for BOS
        self.word2Idx['OOV'] = 0
        self.word2Idx['BOS'] = 1
        self.idx2Word[0] = 'OOV'
        self.idx2Word[1] = 'BOS'

        if os.path.exists(w2v_file):
            begin_time = time.time()
            with open(w2v_file, 'r') as w2v:
                wordIdx = 2
                for line in w2v:
                    embeddingStr = line.split()
                    word = embeddingStr.pop(0)
                    embedding = torch.FloatTensor([float(v_i) for v_i in embeddingStr])
                    self.term_emb[wordIdx] = embedding
                    self.word2Idx[word] = wordIdx
                    self.idx2Word[wordIdx] = word    
                    wordIdx += 1
            # record the size of our vocabulary
            self.nt = wordIdx
            self.term_emb = self.term_emb.narrow(0, 0, self.nt)
            end_time = time.time()
            print "-- Reading word2vec takes %.4f, secs" % round(end_time - begin_time, 5)
        else:
            # the file does not exist
            print "No embeddings of the given dimension," \
            " the filename is %s" % w2v_file
        return

    ## this method reads and creates the nonterminal embeddings
    def create_nt_emb(self):

        nt_file = constants.NON_TERMINAL_EMBEDDING_FILE

        if os.path.exists(nt_file):
            begin_time = time.time()
            with open(nt_file, 'r') as nt:
                nt.next()
                self.dnt = self.nnt = int(nt.next().split('\t', 1)[0]) + 2

                self.nonterm_emb = torch.eye(self.nnt, self.dnt)
                print "The number of nonterminals" \
                "(include symbols U_TM and U_NTM) is %d" % self.nnt

                # Set up the special symbol UNARY for 
                # unary terminal and nonterminal rules:
                # 0 U_TM
                # 1 U_NTM
                # 2 ROOT
                # ...

                self.nonterm2Idx['U_TM'] = 0
                self.nonterm2Idx['U_NTM'] = 1
                self.idx2Nonterm[0] = 'U_TM'
                self.idx2Nonterm[1] = 'U_NTM'
                self.new_nt_num = 2

                for line in nt:
                    mapping = line.split()

                    nonterminal = mapping.pop(0)
                    index = int(mapping.pop(0)) + self.new_nt_num    

                    self.nonterm2Idx[nonterminal] = index
                    self.idx2Nonterm[index] = nonterminal
            end_time = time.time()
            print "-- Reading nonterminals takes %.4f, secs" \
                % round(end_time - begin_time, 5)
        else:
             # the file does not exist
            print "No such nonterminal embedding file," \
                " the filename is %s" % nt_file         
        return

    ##  read corpus
    def read_corpus(self):

        # lexicon = (nt, nnt)
        # unary_dict = (nnt, nnt)
        # binary_dict = (nnt, nnt, nnt)

        self.lexicon = {}
        self.unary = {}
        self.binary = {}

        if os.path.exists(self.file_data):
            with open(self.file_data, 'r') as data:
                self.lines = data.readlines()
                nsen = len(self.lines) / 2
                assert (len(self.lines) % 2 == 0)
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
                            if not tpl in self.binary:
                                self.binary[tpl] = set()
                            self.binary[tpl].add(parent)
                        elif parse[4*j+2] == 'u':
                            parent = int(parse[4*j+1]) + self.new_nt_num
                            child = int(parse[4*j+3]) + self.new_nt_num
                            if not child in self.unary:
                                self.unary[child] = set()
                            self.unary[child].add(parent)
                        elif parse[4*j+2] == 't':
                            parent = int(parse[4*j+1]) + self.new_nt_num
                            try:
                                child = self.word2Idx[parse[4*j+3].lower()]
                            except KeyError:
                                child = 0
                            if not child in self.lexicon:
                                self.lexicon[child] = set()
                            self.lexicon[child].add(parent)
            lexicon = {}
            for s in self.lexicon:
                temp = [x for x in iter(self.lexicon[s])]
                lexicon[s] = temp
            unary = {}
            for s in self.unary:
                temp = [x for x in iter(self.unary[s])]
                unary[s] = temp
            binary = {}
            for s in self.binary:
                temp = [x for x in iter(self.binary[s])]
                binary[s] = temp
            self.lexicon = lexicon
            self.unary = unary
            self.binary = binary        
        else:
            print "No such input file, the filename is %s" % self.file_data
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
                            self.ut_t[senNum-1][i_ut] = self.word2Idx[rest[4*j+3].lower()]
                            self.sens[senNum-1][i_ut] = self.word2Idx[rest[4*j+3].lower()]
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
            print " - Extracting input takes %.4f" % round(end - start, 5)

            return -1 if wrong else senIdx

    def is_digit(self, n):
        try:
            int(n)
            return True
        except ValueError:
            return False

    def print_rules(self):
        print "Lexicon is: "
        for word in self.lexicon:
            for NT in self.lexicon[word]:
                print "%s ---> %s" \
                    % (self.idx2Nonterm[NT], self.idx2Word[word])                   

        print "---------------------------------------------------------------"

        print "Unary nonterminal rules are: "
        dickt = {}
        for child in self.unary:
            dickt[child] = []
            for parent in self.unary[child]:
                dickt[child].append(parent)
                if parent in dickt  and  child in dickt[parent]:
                    print "FUCK WE ARE FUCKED!", parent, " ", child, "###############################################"
                print "%s ---> %s" \
                    % (self.idx2Nonterm[parent], self.idx2Nonterm[child])

        print "---------------------------------------------------------------"

        print "Binary rules are: "
        for key in self.binary:
            for parent in self.binary[key]:
                print "%s ---> %s %s" % (self.idx2Nonterm[parent], self.idx2Nonterm[key[0]], self.idx2Nonterm[key[1]])

    def read_and_process(self):
        if self.read_data:
            if os.path.exists(constants.CORPUS_INFO_FILE):
                os.remove(constants.CORPUS_INFO_FILE)
            print "Reading and processing data ... "
            start = time.time()

            self.read_word2vec()
            self.create_nt_emb()
            self.read_corpus()

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
                    'word2Idx': self.word2Idx,
                    'idx2Word': self.idx2Word,
                    'nonterm2Idx': self.nonterm2Idx,
                    'idx2Nonterm': self.idx2Nonterm,
                    'lexicon': self.lexicon,
                    'unary': self.unary,
                    'binary': self.binary,
                    'lines': self.lines
                }, constants.CORPUS_INFO_FILE)
        else:
            # read existing data, so we don't need to process again
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
            self.word2Idx = d['word2Idx']
            self.idx2Word = d['idx2Word']
            self.nonterm2Idx = d['nonterm2Idx']
            self.idx2Nonterm = d['idx2Nonterm']
            self.lines = d['lines']
            self.lexicon = d['lexicon']
            self.unary = d['unary']
            self.binary = d['binary']
            self.new_nt_num = 2
            end = time.time()
        print "Reading data takes %.4f secs" % round(end - start, 5)

    def get_sen(self, indices):
        return " ".join([self.idx2Word[i] for i in indices])

    def get_idx(self, sen):
        sen_w = ['BOS'] + [w.lower() for w in sen.split()]
        sen_i = torch.LongTensor(1, len(sen_w))
        for i in xrange(len(sen_w)):
            try:
                sen_i[0][i] = self.word2Idx[sen_w[i]]
            except KeyError:
                sen_i[0][i] = 0
        return sen_i
