"""
Processors of data. The class is responsible for creating
terminals and nonterminals' embeddings, making training sets (headified or not),
and some other pre-processing before the actual training of the model.
"""
import ast
import constants
import nltk
import numpy as np
import os
import random
import time
import torch

from collections import defaultdict
from nltk import Tree
from numpy import array
from ptb import ptb
from subprocess import check_call
from unk import signature as sig
from util import oneline, head_binarize, binarize, unbinarize

"""
The processor for the original neural left context model.
"""
class Processor(object):

    def __init__(self, train_file, make_train, read_data, verbose):
        '''
        Initialize the processor, taking inputs from the main class
        @param train_file   The filename of the train file
        @param read_data    Whether to read WSJ and word2vec embedding
        @param make_train   Whether to make new training set (e.g. more sentences to train)
        @param verbose      Whether to have printout for debugging
        '''
        self.train_file = train_file    # the train file
        self.read_data = read_data      # whether to read new data
        self.verbose = verbose          # verbose mode or not
        self.make_train = make_train    # whether to make train set


    #####################################################################################
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


    #####################################################################################
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


    #####################################################################################
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


    #####################################################################################
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


    #####################################################################################
    def make_trainset(self):
        begin_time = time.time()

        train_trees = list(
            ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=10)
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


    #####################################################################################
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




"""
The processor for neural left context model with bilexical information.
(trained on headified trees)
"""
class ProcessorOfHeadedTree(object):

    def __init__(self, train_file, make_train, read_data, verbose):
        '''
        Initialize the processor, taking inputs from the main class
        @param train_file   The filename of the train file
        @param read_data    Whether to read WSJ and word2vec embedding
        @param make_train   Whether to make new training set (e.g. more sentences to train)
        @param verbose      Whether to have printout for debugging
        '''
        self.train_file = train_file    # the train file
        self.read_data = read_data      # whether to read new data
        self.verbose = verbose          # verbose mode or not
        self.make_train = make_train    # whether to make train set


    #####################################################################################
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


    #####################################################################################
    def init_word_indices(self):
        '''
        Temporarily we will train our own word embeddings.
        According to the experiment result, it shows that training
        a new terminal embedding is no worse than using existing word2vec embedding;
        but it is faster than using the word2vec. Because the normal word2vec embedding
        has at least 100 dimensions (300 if it is case sensitive). And this becomes slow.
        
        Also, for prediction, originally, we want something like:

        Let E be the terminal embedding, dim(E) = num_words * word_dim
            W be some weight matrix,     dim(W) = word_dim * conditions_dim

        P( book | conditions ) = softmax(E*W*conditions).select(index_of_book)

        But after some experiment, it shows that it is equally good if we just simply:

        P( book | conditions ) = softmax( g1(relu( g2(conditions) )) ).select(index_of_book)
        So basically we just pass the conditions to a single layer neural net and softmax directly.

        In this way, it becomes possible to use neural net, however in the original case the computation
        is very intensive

        '''

        begin_time = time.time()

        # all_trees are all the trees we can use in training -- (trainset) from the WSJ 2-21
        all_trees = list(ptb("train"))

        # here we follow the old convention from Berkeley's to record the frequency
        # of each word. But we simplify the process such that we only have one threshold, namely
        # the RARE_TRESHOLD (10). There are about 7000+ words that have appeared more than 10
        # times in the WSJ 2-21. And we create signatures for those that do not.
        # Details for creating signature are in Signature in util.py
        self.word_frequency = defaultdict(int)

        for (sentence, _) in all_trees:
            sen_split = sentence.strip().split()
            for word in sen_split:
                self.word_frequency[word] += 1

        # the dimension of terminals -- self.dt
        self.dt = 80
        self.w2idx = {}
        self.idx2w = []
        # there's a special symbol here called BOS, we add this symbol
        # so that we have "some left context" even at the very beginning
        # of the sentence
        self.w2idx[constants.BOS] = 0
        self.idx2w.append(constants.BOS)

        # the number of terminals -- self.nt
        self.nt = 1

        oov_set = set()

        for word, frequency in self.word_frequency.iteritems():
            if frequency < constants.RARE_THRESHOLD:
                _, signature = sig(word, 0)
                oov_set.add(signature)
            else:
                self.w2idx[word] = self.nt
                self.idx2w.append(word)
                self.nt += 1

        # we also want to include all signatures that we haven't covered in
        # training set.
        rest_trees = list(ptb("dev", "test"))

        for (sentence, _) in rest_trees:
            sen_split = sentence.strip().split()
            for word in sen_split:
                if word not in self.word_frequency:
                    _, signature = sig(word, 0)
                    oov_set.add(signature)

        for oov in oov_set:
            self.w2idx[oov] = self.nt
            self.idx2w.append(oov)
            self.nt += 1

        end_time = time.time()

        if self.verbose:
            print "-- Initializing word indices takes %.4f s" % round(end_time - begin_time, 5)
            print "   # words: ", self.nt
        return

    #####################################################################################
    def init_nonterminal_indices(self):
        '''
        Temporarily we will train our own nonterminal embeddings.
        The reason is roughly the same as in the terminal case.
        Also, by doing this, the nonterminal embedding is no longer 1-hot, actually
        after some experiment I found that this is better than 1-hot embedding.

        Read nonterminals from the nonterminal embedding file (there's no embedding,
        only a list of nonterminals --level 0 nonterminals)
        '''
        if not self.check_file_exists(constants.NT_EMB_FILE, "nonterminal embeddings"):
            return

        # dimension of the nonterminals -- self.dnt
        self.dnt = 30
        begin_time = time.time()
        with open(constants.NT_EMB_FILE, 'r') as nt_f:
            nt_f.next()                                 # skip the comment

            # There is a special symbol called TERMINAL that is used in 
            # unary chain prediction later.
            # According to the model, any binary expansion is followed by
            # unary expansion on each newly expanded child, thought the expansion
            # could be trivial (in this case, it means no expansion).
            # Therefore, at the leaf, the unary expansion ends up in the TERMINAL symbol.

            self.nnt = int(nt_f.next()) + 1  # +1 for TERMINAL

            if self.verbose:
                print "# nonterminals " \
                "(include symbols TERMINAL): %d" % self.nnt

            self.nt2idx = {}
            self.idx2nt = []
            self.nt2idx[constants.TERMINAL] = 0
            self.idx2nt.append(constants.TERMINAL)

            idx = 1
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


    #####################################################################################
    def init_unary_chains(self):
        '''
        Remember all unary chains appeared in the treebank and index them.
        In the model, a whole unary chain is expanded at the same time.
        The only thing we need to care about is which unary chain to expand.
        During parsing, we need a dictionary of lists to record for each bottom nonterminal B,
        which possible unary chains U we could append upon it to reach a top nonterminal A.
        '''
        all_trees = list(ptb(["train", "dev", "test"]))

        self.idx2u = []
        self.unary = {}
        # number of unary chains
        self.nunary = 0
        # we need to recover the unary chain during print_parse, therefore
        # a simple and clever way would be to store the prefix and suffix of
        # a certain unary chain. For example, for a unary chain:
        #
        #   A -> B -> C -> .... that appears in the middle of a tree
        #   prefix = (A (B 
        #   suffix = ))
        #   so if we have the parse starting at C, called it parse_C
        #   the parse staring at A becomes: prefix + parse_C + suffix
        self.unary_prefix = []
        self.unary_suffix = []
        self.w_U = {}

        for (_, gold_tree) in all_trees:
            head, unary_chain = self.find_unary(gold_tree)
            if len(unary_chain) > 1:
                if unary_chain not in self.idx2u:
                    self.add_unary_chain(unary_chain)

        for ur in self.idx2u:
            reverse = list(reversed(ur))
            #print "-".join([self.idx2nt[x] for x in reverse])
            tmp = reverse[:-1]
            pre = " (".join([self.idx2nt[x] for x in tmp])
            self.unary_prefix.append("(" + pre + " ")
            self.unary_suffix.append(")" * len(tmp))

        print " # unary chains: ", len(self.idx2u)


    def add_unary_chain(self, unary_chain):
        '''
        helper function to add a new unary_chain, used in find_unary. 
        '''
        self.idx2u.append(unary_chain)
        bottom = unary_chain[0]
        top = unary_chain[-1]
        if bottom not in self.unary:
            self.unary[bottom] = []
        self.unary[bottom].append( (self.nunary, top) )
        self.nunary += 1


    def find_unary(self, tree):
        '''
        find all unary chains in a tree.
        @return head   The head of the tree
        @return chain  The unary chain on the top of the tree
        '''

        # consider the current rule A -> B (C), notice that the tree must have 
        # been binarized already
        A = self.nt2idx[tree.label()]

        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]
            return word, [0, A]
        else:
            nchild = 0 # number of child
            for subtree in tree:
                if nchild == 0:
                    headB, B_unary_chain = self.find_unary(subtree)
                else:
                    headC, C_unary_chain = self.find_unary(subtree)
                nchild += 1

            if nchild == 1:
                # unary expansion
                B_unary_chain.append(A)
                return headB, B_unary_chain

            else:
                # binary expansion
                if len(B_unary_chain) > 1 and B_unary_chain not in self.idx2u:
                    self.add_unary_chain(B_unary_chain)

                if len(C_unary_chain) > 1 and C_unary_chain not in self.idx2u:
                    self.add_unary_chain(C_unary_chain)

                # to speed up the parsing step, temporarily we only allow
                # a terminal unary expansion if the it is compatible with
                # the terminal word. Therefore we need a dictionary from
                # word to its possible unary chain above
                if not headB == None:
                    idx = self.get_word_idx(headB)
                    if idx not in self.w_U:
                        self.w_U[idx] = []
                    tpl = (self.idx2u.index(B_unary_chain), B_unary_chain[-1])
                    if tpl not in self.w_U[idx]:
                        self.w_U[idx].append(tpl)

                if not headC == None:
                    idx = self.get_word_idx(headC)
                    if idx not in self.w_U:
                        self.w_U[idx] = []
                    tpl = (self.idx2u.index(C_unary_chain), C_unary_chain[-1])
                    if tpl not in self.w_U[idx]:
                        self.w_U[idx].append(tpl)
                return None, [A]


    #####################################################################################
    def make_trainset(self):
        '''
        Making the trainset. In other word, we create a more compact and
        convenient representation for each tree in our trainset: we store
        separately the binary and unary rules in a tree, and each rule has
        the info of its parent and children as well as the info of positions.
        '''
        begin_time = time.time()

        train_trees = list(
            ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=20000)
        )

        f = open(self.train_file, 'w')
        num_sen = 0
        counter = 0

        self.headify_trees(train_trees)
        self.B_AC = {}

        with open(constants.HEADIFIED_FILE, 'r') as f_head:
            for line in f_head:
                counter += 1
                if counter == 1:    
                    continue
                unbinarized_tree = Tree.fromstring(
                    line.replace("@(", "(^").replace("@", "")
                )

                d = self.encode_head_tree(head_binarize(unbinarized_tree))

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

    def headify_trees(self, train_trees):
        '''
        helper function to headify all train trees. (Thanks to Jason's perl script)
        '''
        f_trees = open(constants.TREES_FILE, 'w')

        for (_, tree) in train_trees:
            f_trees.write(oneline(unbinarize(tree)) + "\n")

        f_trees.close()

        f_head = open(constants.HEADIFIED_FILE, 'w')

        check_call([
            'perl',
            'headify',
            'newmarked.mrk',
            constants.TREES_FILE
        ], stdout=f_head)

        f_head.close()


    def encode_head_tree(self, tree):
        '''
        Encode a head tree to a convenient and simply representation.
        @return dict   The dictionary of binary, unary rules and sentence terminal
                       indices in the tree.
        '''
        self.binary_list = []   # list of binary rules in the tree
        self.unary_list = []    # list of unary rules in the tree
        self.terminal_list = [] # list of terminals (the sentence)

        self.wi = -1 # word index

        _, child, chain, head, _ = self.traverse_head_tree(tree)

        # append the ROOT -> * unary rule
        self.unary_list.append(
            (0, child, self.idx2u.index(chain), head)
        )

        '''
        # DEBUG
        print tree.pretty_print()
        for A, B_i, B, B_head, C_i, C, C_head, Bih in self.binary_list:
            print "{} {} {} -> {}[{}] {}[{}] {}".format(
                B_i, C_i,
                self.idx2nt[A], self.idx2nt[B],  self.idx2w[B_head],
                self.idx2nt[C], self.idx2w[C_head], Bih)

        print ""

        for B_i, B, u, B_head in self.unary_list:
            print B_i, self.idx2nt[B], self.idx2w[B_head], [self.idx2nt[x] for x in self.idx2u[u]]
        '''
        return {"b": self.binary_list, "u": self.unary_list, "s": self.terminal_list}


    def traverse_head_tree(self, tree):
        '''
        Traverse the headified tree to get the desired list of binary and unary rules.
        @return left context position, nonterminal, unary_chain, head word
        '''

        label = tree.label() # the current nonterminal label

        '''
        Let me give an example of a headified tree:
                            S
                            |
                        ----------
                        |         |
                        NP       @S
                        |         |
                       ^NNP   ----------
                              |        |
                              ^VP      .
                              |
                            -----
                            |    |
                            ^V    NP 
        Notice that:
        if a label startswith ^ or @, then it is the head child among all its siblings
        if a label startswith @, it is also the new intermediate nonterminal 
                                 that is introduced because of the binarization
        So when we binarize, we actually are doing the head binarization, meaning that
        the binarized part always include the head child.
        '''

        is_head = False
        if label.startswith('^'):
            label = label[1:]
            is_head = True

        if label.startswith('@'):
            is_head = True

        A = self.nt2idx[label]

        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]
            idx = self.get_word_idx(word)
            self.terminal_list.append(idx)
            self.wi += 1

            return self.wi, A, [0, A], idx, is_head

        else:
            nchild = 0
            # a binary rule A -> B C or unary rule A -> B
            for subtree in tree:

                if nchild == 0:
                    B_i, B, B_unary, B_head, B_is_head = self.traverse_head_tree(subtree)
                else:
                    C_i, C, C_unary, C_head, C_is_head = self.traverse_head_tree(subtree)

                nchild += 1

            if nchild == 1:
                # unary rule
                B_unary.append(A)
                return B_i, A, B_unary, B_head, True

            else:
                # binary rule

                if len(B_unary) > 1:
                    self.unary_list.append(
                        (B_i, B, self.idx2u.index(B_unary), B_head)
                    )

                if len(C_unary) > 1:
                    self.unary_list.append(
                        (C_i, C, self.idx2u.index(C_unary), C_head)
                    )

                self.binary_list.append(
                    (A, B_i, B, B_head, C_i, C, C_head, 0 if B_is_head else 1)
                )

                if B not in self.B_AC:
                    self.B_AC[B] = []
                tpl = (A, C)
                if tpl not in self.B_AC[B]:
                    self.B_AC[B].append(tpl)

                return B_i, A, [A], B_head if B_is_head else C_head, is_head


    def get_word_idx(self, word):
        '''
        Helper function to get index of a word, if it appears frequently, we return
        the index directly, otherwise we return the index of its signature.
        '''
        frequency = self.word_frequency[word]

        if frequency < constants.RARE_THRESHOLD:
            _, word = sig(word, 0)
        return self.w2idx[word]


    #####################################################################################

    def shuffle(self):
        '''
        shuffle the train set.
        '''
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

            self.sens = torch.LongTensor(bsz, m).fill_(0)

            # B is head, @B
            # P( C | A, B, v(h), alpha(A), alpha(C) )
            # P( h' | C, v(h), alpha(C) )

            self.B_A = []
            self.B_B = []
            self.B_C = []
            self.B_BH = []
            self.B_CH = []
            self.B_Bi = []
            self.B_Ci = []

            # C is head, @C
            # P( h' | B, v(h), alpha(A) )
            # P( C | A, B, v(h), v(h'), alpha(A), alpha(C) )
            self.C_A = []
            self.C_B = []
            self.C_C = []
            self.C_BH = []
            self.C_CH = []
            self.C_Bi = []
            self.C_Ci = []

            # P( unary | A, alpha(A), v(h))
            self.U = []
            self.U_A = []
            self.U_Ai = []
            self.U_H = []

            num_sen = 0

            while num_sen < bsz and idx < self.trainset_length:
                d = ast.literal_eval(self.train_data[idx])
                sentence = d['s']
                bl = d['b']
                ul = d['u']
 
                # get the encoded sentence, exclude the last word
                # since we only need left context
                for i in xrange(1, m):
                    if i >= len(sentence):
                        break
                    self.sens[num_sen,i] = sentence[i-1]

                previous = num_sen * m

                # binary
                for A, B_i, B, B_head, C_i, C, C_head, which in bl:
                    if which: # C is head
                        self.C_A.append(A)
                        self.C_B.append(B)
                        self.C_C.append(C)
                        self.C_BH.append(B_head)
                        self.C_CH.append(C_head)
                        self.C_Bi.append(B_i+previous)
                        self.C_Ci.append(C_i+previous)
                    else: # B is head
                        self.B_A.append(A)
                        self.B_B.append(B)
                        self.B_C.append(C)
                        self.B_BH.append(B_head)
                        self.B_CH.append(C_head)
                        self.B_Bi.append(B_i+previous)
                        self.B_Ci.append(C_i+previous)

                # unary
                for A_i, A, index, head in ul:
                    self.U.append(index)
                    self.U_A.append(A)
                    self.U_Ai.append(A_i+previous)
                    self.U_H.append(head)

                num_sen += 1
                idx += 1
            '''
            # DEBUG

            for i in xrange(len(self.B_A)):
                print " # {}, {} #: RULE {} ({}) -> {} {} ({})".format(
                    self.B_Bi[i], self.B_Ci[i],
                    self.idx2nt[self.B_A[i]], self.idx2w[self.B_BH[i]],
                    self.idx2nt[self.B_B[i]], 
                    self.idx2nt[self.B_C[i]], self.idx2w[self.B_CH[i]]
                )

            print "-" * 10

            for i in xrange(len(self.C_A)):
                print " # {}, {} #: RULE {} ({}) -> {} ({}) {}".format(
                    self.C_Bi[i], self.C_Ci[i],
                    self.idx2nt[self.C_A[i]], self.idx2w[self.C_CH[i]],
                    self.idx2nt[self.C_B[i]], self.idx2w[self.C_BH[i]],
                    self.idx2nt[self.C_C[i]]
                )

            print "-" * 10

            for i in xrange(len(self.U)):
                print self.U_Ai[i], self.idx2nt[self.U_A[i]], " -> ", self.U[i], self.idx2w[self.U_H[i]]
            '''
            return idx if idx < self.trainset_length else -1

    def read_train_data(self):
        if not self.check_file_exists(self.train_file, "training data"):
            return

        with open(self.train_file, 'r') as data:
            self.train_data = data.readlines()

    def process_data(self):
        '''
        This is the main function of processor. Other class will only
        call this function. Based on whether the processor needs to
        read_data (create all things from scratch), or make_train (only
        enlarge the trainset, but leaves embeddings the same), or do
        nothing but read in the data from an existing CORPUS_INFO_FILE
        '''
        if self.read_data:
            if os.path.exists(constants.CORPUS_INFO_FILE):
                os.remove(constants.CORPUS_INFO_FILE)

            if self.verbose:
                print "Reading and processing data ... "

            start = time.time()

            self.init_word_indices()
            #self.init_word_embeddings()
            self.init_nonterminal_indices()
            self.init_unary_chains()

            self.make_trainset()
            self.read_train_data()
            self.trainset_length = len(self.train_data)

            end = time.time()

            # DEBUG: self.print_rules()

            # save those for future use
            torch.save({
                    'trainset_length': self.trainset_length,

                    'nt': self.nt,
                    'dt': self.dt,
                    'nnt': self.nnt,
                    'dnt': self.dnt,

                    'nunary': self.nunary,
                    'unary': self.unary,
                    'unary_prefix': self.unary_prefix,
                    'unary_suffix': self.unary_suffix,
                    'idx2u': self.idx2u,

                    'B_AC': self.B_AC,
                    'w_U': self.w_U,
                    #'word_emb': self.word_emb,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'word_frequency': self.word_frequency,

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

            self.nt = d['nt']
            self.dt = d['dt']
            self.nnt = d['nnt']
            self.dnt = d['dnt']

            self.nunary = d['nunary']
            self.unary = d['unary']
            self.unary_prefix = d['unary_prefix']
            self.unary_suffix = d['unary_suffix']
            self.idx2u = d['idx2u']

            self.w_U = d['w_U']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            #self.word_emb = d['word_emb']

            self.word_frequency = d['word_frequency']

            self.make_trainset()
            self.read_train_data()
            self.trainset_length = len(self.train_data)

            os.remove(constants.CORPUS_INFO_FILE)

            torch.save({
                    'trainset_length': self.trainset_length,

                    'nt': self.nt,
                    'dt': self.dt,
                    'nnt': self.nnt,
                    'dnt': self.dnt,

                    'nunary': self.nunary,
                    'unary': self.unary,
                    'unary_prefix': self.unary_prefix,
                    'unary_suffix': self.unary_suffix,
                    'idx2u': self.idx2u,

                    'B_AC': self.B_AC,
                    'w_U': self.w_U,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'word_frequency': self.word_frequency,

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

            self.nt = d['nt']
            self.dt = d['dt']
            self.nnt = d['nnt']
            self.dnt = d['dnt']

            self.nunary = d['nunary']
            self.unary = d['unary']
            self.unary_prefix = d['unary_prefix']
            self.unary_suffix = d['unary_suffix']
            self.idx2u = d['idx2u']

            self.B_AC = d['B_AC']
            self.w_U = d['w_U']
            #self.word_emb = d['word_emb']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            self.word_frequency = d['word_frequency']

            self.train_data = d['train_data']

            end = time.time()

        if self.verbose:
            print "Reading data takes %.4f secs" % round(end - start, 5)
 
    def get_idx(self, sen):
        '''
        @return sen_i   a torch tensor with each element being the index of
                        the word in sentence. Append BOS at the beginning 
                        (w2idx['BOS'] = 0)
        '''
        sen_w = sen.split()
        length = len(sen_w)

        sen_i = torch.LongTensor(length+1)
        sen_i[0] = 0
        for i in xrange(1, length+1):
            word = sen_w[i-1]
            sen_i[i] = self.get_word_idx(word)
        return sen_i



