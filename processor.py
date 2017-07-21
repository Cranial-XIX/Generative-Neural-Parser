import ast
import constants
import nltk
import numpy as np
import os
import random
import time
import torch

from collections import defaultdict
from numpy import array
from ptb import ptb
from unk import signature as sig

class Processor(object):  

    def __init__(self, train_file, make_train, read_data, verbose):
        self.train_file = train_file    # the train file
        self.read_data = read_data      # whether to read new data
        self.verbose = verbose          # verbose mode or not
        self.make_train = make_train    # whether to make train set

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

    def init_word_embeddings(self):
        '''
        Read in GloVe word embeddings.
        '''
        if not self.check_file_exists(constants.GLOVE_FILE, "GloVe embeddings"):
            return

        begin_time = time.time()

        all_trees = list(ptb("train"))
        self.word_frequency = defaultdict(int)

        for (sentence, _) in all_trees:
            sen_split = sentence.strip().split()
            for word in sen_split:
                self.word_frequency[word] += 1

        glove = {}
        with open(constants.GLOVE_FILE, 'r') as f:
            for line in f:
                emb = line.split()
                word = emb.pop(0)
                glove[word] = torch.FloatTensor(map(float, emb))

        self.w2idx = {}
        self.idx2w = []
        self.w2idx[constants.BOS] = 0
        self.idx2w.append(constants.BOS)

        self.nt = 1
        oov_set = set()

        print " # total words in WSJ : ", len(self.word_frequency)

        counter = 0
        for word, frequency in self.word_frequency.iteritems():
            if word not in glove and frequency > constants.RARE_THRESHOLD:
                counter += 1

        self.dt = constants.DIM_TERMINAL + counter
        self.word_emb = torch.zeros(len(glove), self.dt)

        counter = -1
        for word, frequency in self.word_frequency.iteritems():
            has_word = word in glove
            if frequency > constants.UNCOMMON_THRESHOLD and has_word:
                self.w2idx[word] = self.nt
                self.idx2w.append(word)
                self.word_emb[self.nt][:300] = glove[word]
                self.nt += 1
            elif frequency > constants.RARE_THRESHOLD and has_word:
                self.w2idx[word] = self.nt
                self.idx2w.append(word)
                self.word_emb[self.nt][:300] = glove[word]
                self.word_emb[self.nt][300:314] = sig(word, frequency)[0]
                self.nt += 1
            elif not has_word and frequency > constants.RARE_THRESHOLD:
                counter += 1
                self.w2idx[word] = self.nt
                self.idx2w.append(word)
                self.word_emb[self.nt][300:314] = sig(word, frequency)[0]
                self.word_emb[self.nt][314+counter] = 1
                self.nt += 1
            else:
                emb, signature = sig(word, 0)
                if signature not in oov_set:
                    self.w2idx[signature] = self.nt
                    self.idx2w.append(signature)
                    self.word_emb[self.nt][300:314] = emb
                    oov_set.add(signature)
                    self.nt += 1

        rest_trees = list(ptb(["dev", "test"]))
        for (sentence, _) in rest_trees:
            sen_split = sentence.strip().split()
            for word in sen_split:
                if word not in self.word_frequency:
                    emb, signature = sig(word, 0)
                    if signature not in oov_set:
                        self.w2idx[signature] = self.nt
                        self.idx2w.append(signature)
                        self.word_emb[self.nt][300:] = emb
                        oov_set.add(signature)
                        self.nt += 1

        self.word_emb = self.word_emb.narrow(0, 0, self.nt)
        end_time = time.time()

        if self.verbose:
            print "-- Reading GloVe embeddings takes %.4f s" % round(end_time - begin_time, 5)
            print "   # words: ", self.nt
        return

    def init_nonterminal_embeddings(self):
        '''
        Reads and creates the nonterminal embeddings
        @param nt_file the file name of nonterminals
        '''
        if not self.check_file_exists(constants.NT_EMB_FILE, "nonterminal embeddings"):
            return

        begin_time = time.time()
        with open(constants.NT_EMB_FILE, 'r') as nt_f:
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
            self.nt2idx = {}
            self.idx2nt = []
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

    def init_unary_chains_and_lexicon(self):
        all_trees = list(ptb(["train", "dev", "test"]))

        self.idx2u = []
        self.unary = {}
        self.nunary = 0
        self.unary_prefix = []
        self.unary_suffix = []
        self.lexicon = [[] for x in xrange(self.nt)]

        for (_, gold_tree) in all_trees:
            unary_chain = self.find_unary(gold_tree)
            if len(unary_chain) > 1:
                if unary_chain not in self.idx2u:
                    self.add_unary_chain(unary_chain)

        for ur in self.idx2u:
            reverse = list(reversed(ur))
            tmp = reverse[:-1]
            pre = " (".join([self.idx2nt[x] for x in tmp])
            self.unary_prefix.append("(" + pre + " ")
            self.unary_suffix.append(")" * len(tmp))

    def add_unary_chain(self, unary_chain):
        self.idx2u.append(unary_chain)
        bottom = unary_chain[0]
        if bottom not in self.unary:
            self.unary[bottom] = []
        self.unary[bottom].append(self.nunary)
        self.nunary += 1

    def find_unary(self, tree):
        current_nonterminal = self.nt2idx[tree.label()]
        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]
            word_idx = self.get_word_idx(word)

            if current_nonterminal not in self.lexicon[word_idx]:
                self.lexicon[word_idx].append(current_nonterminal)

            return [current_nonterminal]
        else:
            nchild = 0
            previous = []
            for subtree in tree:
                prev = self.find_unary(subtree)
                previous.append(prev)
                nchild += 1
            if nchild == 1:
                previous[0].append(current_nonterminal)
                return previous[0]
            else:
                left_unary_chain = previous[0]
                right_unary_chain = previous[1]
                if right_unary_chain == None:
                    print current_nonterminal
                if len(left_unary_chain) > 1:
                    if left_unary_chain not in self.idx2u:
                        self.add_unary_chain(left_unary_chain)

                if len(right_unary_chain) > 1:
                    if right_unary_chain not in self.idx2u:
                        self.add_unary_chain(right_unary_chain)

                return [current_nonterminal]

    def make_trainset(self):
        begin_time = time.time()

        train_trees = list(
            ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=2000)
        )

        f = open(self.train_file, 'w')

        num_sen = 0
        counter = 0
        for (sentence, gold_tree) in train_trees:
            counter += 1

            d = self.encode_tree(gold_tree)
            d["s"] = sentence
            if num_sen == 0:
                f.write(str(d))
            else:
                f.write( "\n" + str(d) )
            num_sen += 1

        f.close()
        # DEBUG: print self.convert_tree_to_encoded_list(nltk.Tree.fromstring("(ROOT (S (@S (NP I) (VP (VBP live)))(. .)))"))
        end_time = time.time()

        print "-- Making trainset takes %.4f s" \
            % round(end_time - begin_time, 5)
        print " # sentences ", num_sen

    def encode_tree(self, tree):
        self.binary_list = []
        self.unary_list = []
        self.terminal_list = []
        self.wi = -1 # word index

        position, _, prev = self.traverse_tree(tree)

        # append the ROOT -> * unary rule
        self.unary_list.extend(
            ( position, 2, self.idx2u.index(prev) )
        )

        return {"b": self.binary_list, "u": self.unary_list, "t": self.terminal_list}

    def get_word_idx(self, word):
        frequency = self.word_frequency[word]

        if (frequency <= constants.RARE_THRESHOLD) or (not word in self.w2idx):
            _, word = sig(word, 0)
        return self.w2idx[word]

    def traverse_tree(self, tree):
        current_nonterminal = self.nt2idx[tree.label()]

        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]

            self.wi += 1
            self.terminal_list.extend((self.wi, current_nonterminal, self.get_word_idx(word)))
            return self.wi, current_nonterminal, [current_nonterminal]

        else:
            nchild = 0
            previous = []
            for subtree in tree:
                if nchild == 0:
                    position, child, prev = self.traverse_tree(subtree)
                else:
                    mid, right, prev = self.traverse_tree(subtree)
                previous.append(prev)
                nchild += 1

            if nchild == 1:
                previous[0].append(current_nonterminal)
                return position, current_nonterminal, previous[0]
            else:
                # binary rule
                self.binary_list.extend( (position, mid, current_nonterminal, child, right) )
                nchild = 0
                for prev in previous:
                    head = child if nchild == 0 else right
                    if len(prev) > 1:
                        self.unary_list.extend(
                            ( position, head, self.idx2u.index(prev) )
                        )
                    nchild += 1

                return position, current_nonterminal, [current_nonterminal]

    def shuffle(self):
        np.random.shuffle(self.train_data)

    def next(self, idx, bsz=None):
        '''
        this function extract the next batch of training instances
        and save them for later use
        '''
        if bsz == None:
            ## unsupervised
            print "Not implemented yet"

        else:
            ## supervised

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

            while num_sen < bsz and idx < self.trainset_length:
                d = ast.literal_eval(self.train_data[idx])
                sentence = d['s']
                bl = d['b']
                ul = d['u']
                tl = d['t']
                # get the encoded sentence, exclude the last word
                # since we only need left context
                self.sens[num_sen] = self.get_idx_maxlength(sentence)
                previous = num_sen * m
                
                # binary
                b_zip = zip(bl[0::5], bl[1::5], bl[2::5], bl[3::5], bl[4::5])
                for position, mid, parent, left, right in b_zip:
                    self.p2l[num_p2l] = parent
                    self.p2l_t[num_p2l] = left
                    self.p2l_i[num_p2l] = previous + position
                    num_p2l += 1

                    self.pl2r_p[num_pl2r] = parent
                    self.pl2r_l[num_pl2r] = left
                    self.pl2r_t[num_pl2r] = right
                    self.pl2r_pi[num_pl2r] = previous + position
                    self.pl2r_ci[num_pl2r] = previous + mid
                    num_pl2r += 1

                # unary
                u_zip = zip(ul[0::3], ul[1::3], ul[2::3])
                for position, parent, child in u_zip:
                    self.p2l[num_p2l] = parent
                    self.p2l_t[num_p2l] = 1
                    self.p2l_i[num_p2l] = previous + position
                    num_p2l += 1

                    self.unt[num_unt] = parent
                    self.unt_t[num_unt] = child
                    self.unt_i[num_unt] = previous + position
                    num_unt += 1

                # terminal
                t_zip = zip(tl[0::3], tl[1::3], tl[2::3])
                for position, parent, word in t_zip:
                    self.p2l[num_p2l] = parent
                    self.p2l_t[num_p2l] = 0
                    self.p2l_i[num_p2l] = previous + position
                    num_p2l += 1

                    self.ut[num_ut] = parent
                    self.ut_t[num_ut] = word
                    self.ut_i[num_ut] = previous + position
                    num_ut += 1

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

            return idx if idx < self.trainset_length else -1

    def read_train_data(self):
        if not self.check_file_exists(self.train_file, "training data"):
            return

        with open(self.train_file, 'r') as data:
            self.train_data = data.readlines()

    def process_data(self):
        if self.read_data:
            if os.path.exists(constants.CORPUS_INFO_FILE):
                os.remove(constants.CORPUS_INFO_FILE)

            if self.verbose:
                print "Reading and processing data ... "
            start = time.time()

            self.init_word_embeddings()
            self.init_nonterminal_embeddings()
            self.init_unary_chains_and_lexicon()

            self.make_trainset()
            self.read_train_data()
            self.trainset_length = len(self.train_data)

            end = time.time()

            # DEBUG: self.print_rules()

            # save those for future use
            torch.save({
                    'trainset_length': self.trainset_length,
                    'word_emb': self.word_emb,
                    'nonterm_emb': self.nonterm_emb,

                    'nt': self.nt,
                    'dt': self.dt,
                    'nnt': self.nnt,
                    'dnt': self.dnt,

                    'nunary': self.nunary,
                    'unary': self.unary,
                    'unary_prefix': self.unary_prefix,
                    'unary_suffix': self.unary_suffix,
                    'idx2u': self.idx2u,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'word_frequency': self.word_frequency,
                    'lexicon': self.lexicon,

                    'train_data': self.train_data,
                }, constants.CORPUS_INFO_FILE)
        elif self.make_train:
            start = time.time()
            if self.verbose:
                print "Reading existing data ... "
    
            if not os.path.exists(constants.CORPUS_INFO_FILE):
                print "Error, no corpus info file found"
                return

            d = torch.load(constants.CORPUS_INFO_FILE)
            self.word_emb = d['word_emb']
            self.nonterm_emb = d['nonterm_emb']

            self.nt = d['nt']
            self.dt = d['dt']
            self.nnt = d['nnt']
            self.dnt = d['dnt']

            self.nunary = d['nunary']
            self.unary = d['unary']
            self.unary_prefix = d['unary_prefix']
            self.unary_suffix = d['unary_suffix']
            self.idx2u = d['idx2u']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            self.word_frequency = d['word_frequency']
            self.lexicon = d['lexicon']

            self.make_trainset()
            self.read_train_data()
            self.trainset_length = len(self.train_data)

            os.remove(constants.CORPUS_INFO_FILE)
            torch.save({
                    'trainset_length': self.trainset_length,
                    'word_emb': self.word_emb,
                    'nonterm_emb': self.nonterm_emb,

                    'nt': self.nt,
                    'dt': self.dt,
                    'nnt': self.nnt,
                    'dnt': self.dnt,

                    'nunary': self.nunary,
                    'unary': self.unary,
                    'unary_prefix': self.unary_prefix,
                    'unary_suffix': self.unary_suffix,
                    'idx2u': self.idx2u,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'word_frequency': self.word_frequency,
                    'lexicon': self.lexicon,

                    'train_data': self.train_data,
                }, constants.CORPUS_INFO_FILE)

            end = time.time()

        else:
            # read existing data, so we don't need to process again
            start = time.time()
            if self.verbose:
                print "Reading existing data ... "
    
            if not os.path.exists(constants.CORPUS_INFO_FILE):
                print "Error, no corpus info file found"
                return

            d = torch.load(constants.CORPUS_INFO_FILE)

            self.trainset_length = d['trainset_length']
            self.word_emb = d['word_emb']
            self.nonterm_emb = d['nonterm_emb']

            self.nt = d['nt']
            self.dt = d['dt']
            self.nnt = d['nnt']
            self.dnt = d['dnt']

            self.nunary = d['nunary']
            self.unary = d['unary']
            self.unary_prefix = d['unary_prefix']
            self.unary_suffix = d['unary_suffix']
            self.idx2u = d['idx2u']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            self.word_frequency = d['word_frequency']
            self.lexicon = d['lexicon']

            self.train_data = d['train_data']

            end = time.time()

        if self.verbose:
            print "Reading data takes %.4f secs" % round(end - start, 5)
 
    def get_idx(self, sen):
        sen_w = sen.split()
        length = len(sen_w)

        sen_i = torch.LongTensor(length+1)
        sen_i[0] = 0
        for i in xrange(1, length+1):
            word = sen_w[i-1]
            sen_i[i] = self.get_word_idx(word)
        return sen_i

    def get_idx_maxlength(self, sen):
        sen_w = sen.split()
        length = len(sen_w)

        sen_i = torch.LongTensor(constants.MAX_SEN_LENGTH).fill_(0)
        sen_i[0] = 0
        for i in xrange(1,length):
            word = sen_w[i-1]
            sen_i[i] = self.get_word_idx(word)
        return sen_i

