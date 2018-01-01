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

from nltk import Tree
from numpy import array
from ptb import ptb
from subprocess import check_call
from unk import signature as sig
from util import oneline, head_binarize, binarize, unbinarize

"""
Base Class Processor
"""
class Processor(object):
    def __init__(self, data_folder, make_train, read_data, verbose, seed):
        '''
        Initialize the processor, taking inputs from the main class
        @param data_folder  The directory path of data to be stored
        @param read_data    Whether to read WSJ and word2vec embedding
        @param make_train   Whether to make new training set (e.g. more sentences to train)
        @param verbose      Whether to have printout for debugging
        '''
        self.data_file = data_folder
        self.train_file = data_folder + "train.txt"
        self.dev_file = data_folder + "dev.txt"
        self.test_file = data_folder + "test.txt"

        self.read_data = read_data      # whether to read new data
        self.verbose = verbose          # verbose mode or not
        self.make_train = make_train    # whether to make train set
        np.random.seed(seed)            # set random seed

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

        if not self.check_file_exists(constants.GLOVE_FILE, "GloVe embeddings"):
            return

        begin_time = time.time()

        # all_trees are all the trees we can use in training -- (trainset) from the WSJ 2-21
        train_trees = list(ptb("train"))

        # here we follow the old convention from Berkeley's to record the frequency
        # of each word. But we simplify the process such that we only have one threshold, namely
        # the RARE_TRESHOLD (10). There are about 7000+ words that have appeared more than 10
        # times in the WSJ 2-21. And we create signatures for those that do not.
        # Details for creating signature are in Signature in util.py
        word_frequencies = {}

        num_train_tokens = 0
        num_train_sens = len(train_trees)
        max_length_sen = 0
        for (sentence, _) in train_trees:

            sen_split = sentence.split()
            length = len(sen_split)
            num_train_tokens += length
            if length > max_length_sen:
                max_length_sen = length

            for word in sen_split:
                word = word.rstrip()
                if word in word_frequencies:
                    word_frequencies[word] += 1
                else:
                    word_frequencies[word] = 1

        # the dimension of terminals -- self.dt
        self.dt = 128
        self.w2idx = {}
        self.idx2w = []
        # there's a special symbol here called BOS, we add this symbol
        # so that we have "some left context" even at the very beginning
        # of the sentence
        self.words_set = set()
        self.w2idx[constants.BOS] = 0
        self.idx2w.append(constants.BOS)

        # the number of terminals -- self.nt
        self.nt = 1
        oov_set = set()

        for word, freq in word_frequencies.iteritems():
            if freq <= constants.RARE_THRESHOLD:
                if word.lower() in word_frequencies:
                    knownlc = (word_frequencies[word.lower()] > constants.RARE_THRESHOLD)
                else:
                    knownlc = False
                oov_set.add(sig(word, False, knownlc))
            else:
                self.words_set.add(word)
                self.w2idx[word] = self.nt
                self.idx2w.append(word)
                self.nt += 1

        num_train_oov = len(oov_set)

        print " - In train set: Sequences: {} Tokens: {} Token types: {} " \
            "Unknown types: {} Max sen length: {} ".format(
            num_train_sens, num_train_tokens, self.nt, num_train_oov, max_length_sen)

        # we also want to include all signatures that we haven't covered in
        # training set.
        rest_trees = list(ptb("dev", "test"))

        for (sentence, _) in rest_trees:
            sen_split = sentence.split()
            for word in sen_split:
                word = word.rstrip()
                if word in self.words_set:
                    continue
                knownlc = word.lower() in self.words_set
                oov_set.add(sig(word, False, knownlc))

        for oov in oov_set:
            self.w2idx[oov] = self.nt
            self.idx2w.append(oov)
            self.nt += 1

        self.dt = 300
        self.word_emb = torch.zeros(self.nt, self.dt)

        with open(constants.GLOVE_FILE, 'r') as f:
            for line in f:
                emb = line.split()
                word = emb.pop(0)
                if word in self.w2idx:
                    idx = self.w2idx[word]
                    self.word_emb[idx] = torch.FloatTensor([float(i) for i in emb])

        print " - There are {} number of OOVs. ".format(len(oov_set))

        end_time = time.time()

        if self.verbose:
            print "-- Initializing word indices takes %.4f s" % round(end_time - begin_time, 5)
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
        if not self.check_file_exists(constants.NT_FILE, "nonterminal embeddings"):
            return

        # dimension of the nonterminals -- self.dnt
        self.dnt = 32
        begin_time = time.time()
        with open(constants.NT_FILE, 'r') as nt_f:
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
    def read_input(self):
        if not self.check_file_exists(self.train_file, "training data"):
            return
        if not self.check_file_exists(self.dev_file, "training data"):
            return
        if not self.check_file_exists(self.test_file, "training data"):
            return

        with open(self.train_file, 'r') as data:
            self.train_data = data.readlines()
            self.trainset_length = len(self.train_data)

        with open(self.dev_file, 'r') as data:
            self.dev_data = data.readlines()
            self.devset_length = len(self.dev_data)

        with open(self.test_file, 'r') as data:
            self.test_data = data.readlines()
            self.testset_length = len(self.test_data)


    def shuffle(self):
        '''
        shuffle the train set.
        '''
        np.random.shuffle(self.train_data)


    #####################################################################################
    def init_rules(self):
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

        # the word -> preterminal unary chain dictionary
        self.w_U = {}

        self.B_AC = {}
        self.A2B_bias_mask = torch.FloatTensor(self.nnt, self.nnt).fill_(-10000)

        for (_, gold_tree) in all_trees:
            head, unary_chain = self.find_rules(gold_tree)
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
        helper function to add a new unary_chain, used in find_rules. 
        '''
        self.idx2u.append(unary_chain)
        bottom = unary_chain[0]
        top = unary_chain[-1]
        if bottom not in self.unary:
            self.unary[bottom] = []
        self.unary[bottom].append( (self.nunary, top) )
        self.nunary += 1


    def find_rules(self, tree):
        '''
        find all unary chains in a tree, as well as binary rules
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
                    headB, B_unary_chain = self.find_rules(subtree)
                else:
                    headC, C_unary_chain = self.find_rules(subtree)
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

                B = B_unary_chain[-1]
                C = C_unary_chain[-1]

                # to speed up the parsing step, temporarily we only allow
                # a terminal unary expansion if the it is compatible with
                # the terminal word. Therefore we need a dictionary from
                # word to its possible unary chain above
                if not headB == None:
                    tpl = (self.idx2u.index(B_unary_chain), B, B_unary_chain[1])
                    idx = self.get_word_idx(headB)
                    if idx not in self.w_U:
                        self.w_U[idx] = [tpl]
                    elif tpl not in self.w_U[idx]:
                        self.w_U[idx].append(tpl)

                if not headC == None:
                    tpl = (self.idx2u.index(C_unary_chain), C, C_unary_chain[1])
                    idx = self.get_word_idx(headC)
                    if idx not in self.w_U:
                        self.w_U[idx] = [tpl]
                    elif tpl not in self.w_U[idx]:
                        self.w_U[idx].append(tpl)

                if B not in self.B_AC:
                    self.B_AC[B] = []

                tpl = (A, C)
                if tpl not in self.B_AC[B]:
                    self.A2B_bias_mask[A, B] = 0
                    self.B_AC[B].append(tpl)

                return None, [A]


    #####################################################################################
    def get_word_idx(self, word):
        '''
        Helper function to get index of a word, if it appears frequently, we return
        the index directly, otherwise we return the index of its signature.
        '''

        return self.w2idx[sig(word, word in self.words_set, word.lower() in self.words_set)]

    def is_oov(self, word):
        return not sig(word, word in self.words_set, word.lower() in self.words_set) == word

    def get_idx(self, sentence):
        '''
        @return sen_i   a torch tensor with each element being the index of
                        the word in sentence. Append BOS at the beginning 
                        (w2idx['BOS'] = 0)
        '''
        sen_w = sentence.split()
        length = len(sen_w)

        sen_i = torch.LongTensor(length+1)
        sen_i[0] = 0
        for i in xrange(1, length+1):
            word = sen_w[i-1].rstrip()
            sen_i[i] = self.get_word_idx(word)
        return sen_i


    def make_data(self, dataset, num=None):
        '''
        Making the trainset. In other word, we create a more compact and
        convenient representation for each tree in our trainset: we store
        separately the binary and unary rules in a tree, and each rule has
        the info of its parent and children as well as the info of positions.
        '''
        begin_time = time.time()

        trees = list(
            ptb(dataset, maxlength=constants.MAX_SEN_LENGTH, n=num)
        )

        filename = self.data_file + dataset + ".txt"

        f = open(filename, 'w')
        num_sen = 0
        counter = 0

        for (sentence, gold_tree) in trees:
            counter += 1

            d = self.encode_tree(gold_tree)

            if num_sen == 0:
                f.write(str(d))
            else:
                f.write( "\n" + str(d) )

            num_sen += 1

        f.close()

        end_time = time.time()

        print "-- Making %sset takes %.4f s".format(dataset, round(end_time - begin_time, 5))
        print " # sentences ", num_sen


    def encode_tree(self, tree):
        '''
        Encode a tree to a convenient and simple representation.
        @return dict   The dictionary of binary, unary rules and sentence terminal
                       indices in the tree.
        '''
        self.binary_list = []   # list of binary rules in the tree
        self.unary_list = []    # list of unary rules in the tree
        self.terminal_list = [] # list of terminals (the sentence)

        A, chain = self.traverse_tree(tree)

        # append the ROOT -> * unary rule
        self.unary_list.append(
            ( A, self.idx2u.index(chain) )
        )

        '''
        # DEBUG
        print tree.pretty_print()
        for A, B, C in self.binary_list:
            print "{} -> {} {}".format(
                self.idx2nt[A], self.idx2nt[B], self.idx2nt[C]
            )

        print ""

        for B, u in self.unary_list:
            print self.idx2nt[B], [self.idx2nt[x] for x in self.idx2u[u]]
        '''

        return {"b": self.binary_list, "u": self.unary_list, "s": self.terminal_list}


    def traverse_tree(self, tree):
        '''
        Traverse the tree to get the desired list of binary and unary rules.
        @return nonterminal, unary_chain
        '''

        label = tree.label() # the current nonterminal label

        '''
        Let me give an example a binarized parse tree:
                             S
                             |
                        -----------
                        |         |
                        @S        |
                        |         |
                  -----------     .
                  |         |
                 NNP        VP
                            |
                          ------
                          |    |
                          V    NP

        Notice that except for the model with bilexical info (head binarization),
        we always do left binarization.
        '''

        A = self.nt2idx[label]

        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]
            idx = self.get_word_idx(word)
            self.terminal_list.append( (A, idx) )

            return A, [0, A]

        else:
            nchild = 0
            # a binary rule A -> B C or unary rule A -> B
            for subtree in tree:

                if nchild == 0:
                    B, B_unary = self.traverse_tree(subtree)
                else:
                    C, C_unary = self.traverse_tree(subtree)

                nchild += 1

            if nchild == 1:
                # unary rule
                B_unary.append(A)
                return A, B_unary

            else:
                # binary rule

                if len(B_unary) > 1:
                    self.unary_list.append(
                        ( B, self.idx2u.index(B_unary) )
                    )

                if len(C_unary) > 1:
                    self.unary_list.append(
                        ( C, self.idx2u.index(C_unary) )
                    )

                self.binary_list.append(
                    (A, B, C)
                )

                return A, [A]


    #####################################################################################
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

            # P( A -> B C ) = P( B | A ) * P ( C | A, B )
            self.AA = []
            self.BB = []
            self.CC = []

            # P( unary | A )
            self.U = []
            self.U_A = []

            # P( word | A )
            self.T = []
            self.T_A = []

            num_sen = 0

            while num_sen < bsz and idx < self.trainset_length:
                d = ast.literal_eval(self.train_data[idx])
                tl = d['s']
                bl = d['b']
                ul = d['u']

                # binary
                for A, B, C in bl:
                    self.AA.append(A)
                    self.BB.append(B)
                    self.CC.append(C)

                # unary
                for A, index in ul:
                    self.U.append(index)
                    self.U_A.append(A)

                # lexicon
                for A, word in tl:
                    self.T.append(word)
                    self.T_A.append(A)

                num_sen += 1
                idx += 1

            '''
            # DEBUG

            for i in xrange(len(self.B_A)):
                print " RULE {} -> {} {}".format(
                    self.AA[i],
                    self.BB[i],
                    self.CC[i]
                )

            print "-" * 10

            for i in xrange(len(self.U)):
                print self.idx2nt[self.U_A[i]], " -> ", self.U[i]

            '''
            next_bch = [
                self.AA, self.BB, self.CC,
                self.U, self.U_A,
                self.T, self.T_A
            ]

            if idx >= self.trainset_length:
                idx = -1

            return idx, next_bch


    #####################################################################################
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
            self.init_nonterminal_indices()
            self.init_rules()

            self.make_data('train')
            self.make_data('dev', 500)
            self.make_data('test', 500)

            self.read_input()

            end = time.time()

            # DEBUG: self.print_rules()

            # save those for future use
            torch.save({
                    'trainset_length': self.trainset_length,
                    'devset_length': self.devset_length,
                    'testset_length': self.testset_length,

                    'train_data': self.train_data,
                    'dev_data': self.dev_data,
                    'test_data': self.test_data,

                    'word_emb': self.word_emb,

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
                    'A2B_bias_mask': self.A2B_bias_mask,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'words_set': self.words_set
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
            self.B_AC = d['B_AC']
            self.A2B_bias_mask = d['A2B_bias_mask']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            self.word_emb = d['word_emb']

            self.words_set = d['words_set']

            self.make_data('train')
            self.make_data('dev', 500)
            self.make_data('test', 500)
            self.read_input()

            os.remove(constants.CORPUS_INFO_FILE)

            torch.save({
                    'trainset_length': self.trainset_length,
                    'devset_length': self.devset_length,
                    'testset_length': self.testset_length,

                    'train_data': self.train_data,
                    'dev_data': self.dev_data,
                    'test_data': self.test_data,

                    'word_emb': self.word_emb,

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
                    'A2B_bias_mask': self.A2B_bias_mask,

                    'w2idx': self.w2idx,
                    'idx2w': self.idx2w,
                    'nt2idx': self.nt2idx,
                    'idx2nt': self.idx2nt,

                    'words_set': self.words_set
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
            self.devset_length = d['devset_length']
            self.testset_length = d['testset_length']
            self.train_data = d['train_data']
            self.dev_data = d['dev_data']
            self.test_data = d['test_data']

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
            self.A2B_bias_mask = d['A2B_bias_mask']

            self.w2idx = d['w2idx']
            self.idx2w = d['idx2w']
            self.nt2idx = d['nt2idx']
            self.idx2nt = d['idx2nt']

            self.word_emb = d['word_emb']

            self.words_set = d['words_set']

            end = time.time()

        if self.verbose:
            print "Reading data takes %.4f secs" % round(end - start, 5)










"""
The (P)rocessor of the model with (L)eft context LSTM features and (N)eural network.
"""
class PLN(Processor):

    def __init__(self, *args):
        super(PLN, self).__init__(*args)


    def encode_tree(self, tree):
        '''
        Encode a tree to a convenient and simply representation.
        @return dict   The dictionary of binary, unary rules and sentence terminal
                       indices in the tree.
        '''
        self.binary_list = []   # list of binary rules in the tree
        self.unary_list = []    # list of unary rules in the tree
        self.terminal_list = [] # list of terminals (the sentence)

        self.wi = -1

        Ai, A, chain = self.traverse_tree(tree)

        # append the ROOT -> * unary rule
        self.unary_list.append(
            ( Ai, A, self.idx2u.index(chain) )
        )

        '''
        # DEBUG
        print tree.pretty_print()
        for Bi, Ci, A, B, C in self.binary_list:
            print "{} -> ({}){} ({}){}".format(
                self.idx2nt[A], Bi, self.idx2nt[B], Ci, self.idx2nt[C]
            )

        print ""

        for Bi, B, u in self.unary_list:
            print Bi, self.idx2nt[B], [self.idx2nt[x] for x in self.idx2u[u]]
        '''

        return {"b": self.binary_list, "u": self.unary_list, "s": self.terminal_list}


    def traverse_tree(self, tree):
        '''
        Traverse the tree to get the desired list of binary and unary rules.
        @return nonterminal, unary_chain
        '''

        label = tree.label() # the current nonterminal label

        '''
        Let me give an example a binarized parse tree:
                             S
                             |
                        -----------
                        |         |
                        @S        |
                        |         |
                  -----------     .
                  |         |
                 NNP        VP
                            |
                          ------
                          |    |
                          V    NP

        Notice that except for the model with bilexical info (head binarization),
        we always do left binarization.
        '''

        A = self.nt2idx[label]

        if tree.height() == 2:
            # is leaf
            word = tree.leaves()[0]
            idx = self.get_word_idx(word)
            self.wi += 1
            self.terminal_list.append( (self.wi, A, idx) )

            return self.wi, A, [0, A]

        else:
            nchild = 0
            # a binary rule A -> B C or unary rule A -> B
            for subtree in tree:

                if nchild == 0:
                    Bi, B, B_unary = self.traverse_tree(subtree)
                else:
                    Ci, C, C_unary = self.traverse_tree(subtree)

                nchild += 1

            if nchild == 1:
                # unary rule
                B_unary.append(A)
                return Bi, A, B_unary

            else:
                # binary rule

                if len(B_unary) > 1:
                    self.unary_list.append(
                        ( Bi, B, self.idx2u.index(B_unary) )
                    )

                if len(C_unary) > 1:
                    self.unary_list.append(
                        ( Ci, C, self.idx2u.index(C_unary) )
                    )

                self.binary_list.append(
                    (Bi, Ci, A, B, C)
                )

                return Bi, A, [A]


    #####################################################################################
    def next_lm(self, idx, bsz=None, dataset='train'):
        '''
        this function extract the next batch of training instances
        and save them for later use
        '''
        if dataset == 'train':
            ds = self.train_data
            length = self.trainset_length
        elif dataset == 'dev':
            ds = self.dev_data
            length = self.devset_length
        else:
            ds = self.test_data
            length = self.testset_length

        ## supervised
        # bsz is batch size, the number of sentences we process each time
        # the maximum number of training instances in a batch
        m = constants.MAX_SEN_LENGTH
        sens = torch.LongTensor(bsz, m).fill_(0)
        targets = []
        targets_I = []
        num_sen = 0

        while num_sen < bsz and idx < length:
            d = ast.literal_eval(ds[idx])
            tl = d['s']
            previous = num_sen * m
            # lexicon
            index = 0
            for Ti, A, word in tl:
                index += 1
                assert Ti == index-1
                targets.append(word)
                targets_I.append(previous + Ti)
                if index < m:
                    sens[num_sen][index] = word

            num_sen += 1
            idx += 1

        next_bch = [
            sens, targets, targets_I
        ]

        if idx >= length:
            idx = -1

        return idx, next_bch
                  
    def next(self, idx, bsz=None, dataset='train'):
        '''
        this function extract the next batch of training instances
        and save them for later use
        '''
        if dataset == 'train':
            ds = self.train_data
            length = self.trainset_length
        elif dataset == 'dev':
            ds = self.dev_data
            length = self.devset_length
        else:
            ds = self.test_data
            length = self.testset_length

        ## supervised
        # bsz is batch size, the number of sentences we process each time
        # the maximum number of training instances in a batch
        m = constants.MAX_SEN_LENGTH
        self.sens = torch.LongTensor(bsz, m).fill_(0)

        # P( A -> B C ) = P( B | A ) * P ( C | A, B )
        self.AA = []
        self.BB = []
        self.CC = []
        self.BI = []
        self.CI = []

        # P( unary | A )
        self.U = []
        self.U_A = []
        self.UI = []

        # P( word | A )
        self.T = []
        self.T_A = []
        self.TI = []

        num_sen = 0

        while num_sen < bsz and idx < length:
            d = ast.literal_eval(ds[idx])
            tl = d['s']
            bl = d['b']
            ul = d['u']
            previous = num_sen * m
            # binary
            for Bi, Ci, A, B, C in bl:
                self.AA.append(A)
                self.BB.append(B)
                self.CC.append(C)
                self.BI.append(Bi+previous)
                self.CI.append(Ci+previous)

            # unary
            for Ai, A, index in ul:
                self.U.append(index)
                self.U_A.append(A)
                self.UI.append(Ai+previous)

            # lexicon
            index = 0
            for Ti, A, word in tl:
                self.T.append(word)
                self.T_A.append(A)
                self.TI.append(Ti+previous)
                index += 1
                if index < m:
                    self.sens[num_sen][index] = word

            num_sen += 1
            idx += 1

        '''
        # DEBUG

        for i in xrange(len(self.AA)):
            print " RULE {} -> ({}){} ({}){}".format(
                self.AA[i],
                self.BI[i],
                self.BB[i],
                self.CI[i],
                self.CC[i]
            )

        print "-" * 10

        for i in xrange(len(self.U)):
            print self.idx2nt[self.U_A[i]], " -> ", self.idx2u[self.U[i]]

        print "_" * 10

        for i in xrange(len(self.T)):
            print self.idx2nt[self.T_A[i]], " -> ", self.idx2w[self.T[i]]
        '''

        next_bch = [
            self.sens,
            self.BI, self.CI, self.AA, self.BB, self.CC,
            self.UI, self.U, self.U_A,
            self.TI, self.T, self.T_A
        ]

        if idx >= length:
            idx = -1

        return idx, next_bch


"""
The (P)rocessor of the model with (B)ilexical information, (L)eft context LSTM features
and (N)eural network.
"""
class PBLN(Processor):

    def __init__(self, *args):
        super(PBLN, self).__init__(*args)


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
            ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=2000)
        )

        f = open(self.train_file, 'w')
        num_sen = 0
        counter = 0

        self.headify_trees(train_trees)

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


    #####################################################################################
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

            next_bch = [
                self.sens,
                self.B_A, self.B_B, self.B_C, self.B_BH, self.B_CH, self.B_Bi, self.B_Ci,
                self.C_A, self.C_B, self.C_C, self.C_BH, self.C_CH, self.C_Bi, self.C_Ci,
                self.U, self.U_A, self.U_Ai, self.U_H
            ]

            if idx >= self.trainset_length:
                idx = -1

            return idx, next_bch

