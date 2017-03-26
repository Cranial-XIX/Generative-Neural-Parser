import numpy as np
from numpy import array
import os
import time
import torch

import constants

class Processor(object):  

    def __init__(self):
        self.num_sen = 0
        self.process_step = -1            

        self.dim_model = constants.DIMENSION_OF_MODEL   # dimension of the model (for now, assume same as word embedding)   

        ## Terminals
        self.dim_term = constants.DIMENSION_OF_MODEL    # word vector dimension, as in word2vec
        self.num_term = -1                              # number of terminals
        self.emb_term_dict = {}                         # embeddings of words        (int -> vector)
        self.word2Idx = {}                              # (string -> int)
        self.idx2Word = {}                              # (int -> string)

        ## Nonterminals
        self.dim_non_term = -1                          # nonterminal feature dimension
        self.num_non_term = -1                          # number of nonterminals
        self.emb_non_term_dict = {}                     # embeddings of nonterminals (int -> vector)
        self.nonterm2Idx = {}                           # (string -> int)
        self.idx2Nonterm = {}                           # (int -> string)

        ## Rules                                        # defined as lists for convenience
        self.unary_dict = []                            # dictionary for unary terminal and nonterminal rules
        self.binary_dict = []                           # dictionary for binary nonterminal rules 

        ## Inputs for model
        self.seq_input_lists = []                       # input sentences ([int])

        self.seq_preterms_lists = []                    # input list of preterminals (using embeddings)
        self.seq_p2l_lists = []                         # list of parent to left sequence
        self.seq_pl2r_lists = []                        # list of parent left to right sequence
        self.seq_u_ntm_lists = []                       # list of unary non terminal sequence

                                                        # used for local MRF
        self.seq_p2l_target_list = []                   # list of p2l targets
        self.seq_pl2r_target_list = []                  # list of pl2r targets
        self.seq_u_ntm_target_list = []                 # list of unary nonterminals targets

        self.variables = [
            self.dim_model,
            self.num_term,
            self.emb_term_dict,
            self.word2Idx,
            self.idx2Word,

            self.dim_non_term,
            self.num_non_term,
            self.emb_non_term_dict,
            self.nonterm2Idx,
            self.idx2Nonterm,

            self.unary_dict,
            self.binary_dict
        ]

    ## This method reads the word embeddings, currently good
    def read_in_word_embeddings(self, dimension):
        '''
        Read in word embeddings from the Word2vec file and store them into a hash map.
        '''
        
        # Get the filename
        word2vecFileName = "%s%d%s"  % (constants.WORD_EMBEDDING_FILE_PREFIX, 
            dimension, constants.WORD_EMBEDDING_FILE_SUFFIX)

        if os.path.exists(word2vecFileName):
            with open(word2vecFileName, 'r') as word2vecFile:
                wordIdx = 0
                for line in word2vecFile:
                    embeddingStr = line.split()
                    word = embeddingStr.pop(0)
                    embedding = [float(v_i) for v_i in embeddingStr]
                    assert(len(embedding) == self.dim_model)
                    self.emb_term_dict[wordIdx] = embedding    
                    self.word2Idx[word] = wordIdx
                    self.idx2Word[wordIdx] = word    
                    wordIdx += 1
            # record the size of our vocabulary
            self.num_term = wordIdx
            # create the embeddings as matrix for the whole vocabulary
            self.emb_term = torch.FloatTensor(self.num_term, dimension)
            for term in xrange(self.num_term):
                for feat_idx in xrange(dimension):
                    self.emb_term[term][feat_idx] = self.emb_term_dict[term][feat_idx]
        else:
            # the file does not exist
            print "No embeddings of the given dimension, the filename is %s" % word2vecFileName
        return

    ## this method reads and creates the nonterminal embeddings
    def create_nonterm_embeddings(self):
        '''
        Read in the nonterminal mappings from file and store them into a hash map.
        '''

        # Get the file name
        mappingFileName = constants.NON_TERMINAL_EMBEDDING_FILE

        if os.path.exists(mappingFileName):
            with open(mappingFileName, 'r') as mappingFile:
                mappingFile.next()
                # currently the two are the same, but we can change this later. TODO(@Bo)
                self.dim_non_term = self.num_non_term = int(mappingFile.next().split('\t', 1)[0]) + 2

                print "The number of nonterminals (include symbols U_TM and U_NTM) is %d" % self.num_non_term

                # Set up the special symbol UNARY for unary terminal and nonterminal rules
                # 0 U_TM
                # 1 U_NTM
                # 2 ROOT
                # ...

                featureVec = [0. for _ in xrange(self.dim_non_term)]
                featureVec[0] = 1.0
                self.emb_non_term_dict[0] = featureVec
                self.nonterm2Idx['U_TM'] = 0
                self.idx2Nonterm[0] = 'U_TM'

                featureVec = [0. for _ in xrange(self.dim_non_term)]
                featureVec[1] = 1.0
                self.emb_non_term_dict[1] = featureVec
                self.nonterm2Idx['U_NTM'] = 1
                self.idx2Nonterm[1] = 'U_NTM'

                self.new_nt_num = 2

                for line in mappingFile:
                    mapping = line.split()
                    nonterminal = mapping.pop(0)

                    index = int(mapping.pop(0)) + self.new_nt_num
                    mapping[0] = mapping[0][1:] # remove '[' symbol at the front
                    mapping[len(mapping)-1] = mapping[len(mapping)-1][:-1] # remove ']' symbol at the back
                    
                    featureVec = [0. for _ in xrange(self.dim_non_term)]
                    for x in mapping:
                        featureVec[int(x)+self.new_nt_num] = 1.0

                    self.emb_non_term_dict[index] = featureVec
                    self.nonterm2Idx[nonterminal] = index
                    self.idx2Nonterm[index] = nonterminal
            self.emb_non_term = torch.FloatTensor(self.num_non_term, self.dim_non_term)
            for nonterm in xrange(self.num_non_term):
                for feat_idx in xrange(self.dim_non_term):
                    self.emb_non_term[nonterm][feat_idx] = self.emb_non_term_dict[nonterm][feat_idx]
        else:
             # the file does not exist
            print "No such nonterminal embedding file, the filename is %s" % mappingFileName           
        return

    def construct_seq_feats_lists(self):
        '''
        Construct two lists of tree matrices based on input files in specified directory treeToMatrixDir
        '''

        # lexicon = (num_term * num_non_term)
        # unary_dict = (num_non_term * num_non_term)
        # binary_dict = (num_non_term * num_non_term * num_non_term)
        #
        self.lexicon = [[False for j in xrange(self.num_non_term)] for i in xrange(self.num_term)]
        self.unary_dict = [[False for j in xrange(self.num_non_term)] for i in xrange(self.num_non_term)]
        self.binary_dict = [[[False for k in xrange(self.num_non_term)] for j in xrange(self.num_non_term)] for i in xrange(self.num_non_term)]

        treeToMatrixDir = constants.TREE_TO_MATRIX_DIR
        print "TreeToMatrix directory is %s" % treeToMatrixDir

        for filename in os.listdir(treeToMatrixDir):
            with open(treeToMatrixDir +  '/' + filename, 'r') as file:

                print "the tree filename is %s" % filename

                while True:
                    # TODO (@Vin) for some reason, we actually need the transpose of the original matrix,
                    # so maybe we want to change it later, for now, I just use the matrix you give me since
                    # I think it might be easier to create matrices in that fashion?
                    try:
                        width, height = [int(num) for num in file.next().split()[:2]]
                    except EOFError:
                        break

                    # skip those special cases
                    if width > constants.MAX_SEN_LENGTH or height > constants.MAX_SEN_HEIGHT:
                        continue
                    self.num_sen += 1

                    input_sentence = []

                    # the initialization
                    preterms = []
                    
                    p2l_matrix = [[[0. for k in xrange(self.dim_non_term)] for j in xrange(width)] for i in xrange(height)]
                    p2l_target = [[-1 for j in xrange(width)] for i in xrange(height)]

                    pl2r_matrix = [[[0. for k in xrange(self.dim_non_term * 2)] for j in xrange(width)] for i in xrange(height)]
                    pl2r_target = [[-1 for j in xrange(width)] for i in xrange(height)]

                    u_ntm_matrix = [[[0. for k in xrange(self.dim_non_term)] for j in xrange(width)] for i in xrange(height)]
                    u_ntm_target = [[-1 for j in xrange(width)] for i in xrange(height)]

                    # copy matrix from file to local var
                    for j in xrange(width):
                        row = file.next().strip().split('\t')

                        for i, rowEntry in enumerate(row):
                            row_splitted = rowEntry.split()
                            parent, leftSib = row_splitted[0], row_splitted[1]
                            rightChild =  ' '.join(row_splitted[2:])

                            if rightChild == "-1":
                                break

                            is_terminal = not self.is_digit(rightChild)

                            ## Append input and preterminal sequence ------------------------

                            if is_terminal:
                                preterms.append(self.emb_non_term_dict[int(parent) + self.new_nt_num])
                                input_sentence.append(self.word2Idx[rightChild.lower()])

                            ## Fill the feats matrix and target matrix ----------------------

                            # parent to vector
                            parentVec = self.emb_non_term_dict[int(parent) + self.new_nt_num]

                            if leftSib == "null":
                                # unary rule
                                p2l_matrix[i][j] = parentVec[:]
                                if is_terminal:
                                    p2l_target[i][j] = 0
                                else:
                                    p2l_target[i][j] = 1
                                    u_ntm_matrix[i][j] = parentVec[:]
                                    u_ntm_target[i][j] = int(rightChild) + self.new_nt_num
                            elif leftSib == "left":
                                # left part of binary rule
                                p2l_matrix[i][j] = parentVec[:]
                                p2l_target[i][j] = int(rightChild) + self.new_nt_num
                            else:
                                pl2r_matrix[i][j] = parentVec[:]
                                pl2r_matrix[i][j].extend(
                                    self.emb_non_term_dict[int(leftSib) + self.new_nt_num])
                                pl2r_target[i][j] = int(rightChild) + self.new_nt_num

                            ## Store rules --------------------------------------------------

                            if is_terminal:
                                # unary terminal rule
                                self.lexicon[int(self.word2Idx[rightChild.lower()])][int(parent) + self.new_nt_num] = True
                            elif leftSib == "null":
                                # unary nonterminal rule
                                self.unary_dict[int(rightChild) + self.new_nt_num][int(parent) + self.new_nt_num] = True
                            elif not leftSib == "left":
                                # binary rule
                                self.binary_dict[int(leftSib) + self.new_nt_num][int(rightChild) + self.new_nt_num][int(parent) + self.new_nt_num] = True

                    file.next()
                    self.seq_input_lists.append(input_sentence)
                    self.seq_preterms_lists.append(preterms)

                    self.seq_p2l_lists.append(p2l_matrix)
                    self.seq_pl2r_lists.append(pl2r_matrix)
                    self.seq_u_ntm_lists.append(u_ntm_matrix)

                    self.seq_p2l_target_list.append(p2l_target)
                    self.seq_pl2r_target_list.append(pl2r_target)
                    self.seq_u_ntm_target_list.append(u_ntm_target)
        return

    def data(self):
        self.torch_inp = torch.LongTensor(self.seq_input_lists)
        self.torch_pre = torch.FloatTensor(self.seq_preterms_lists)
        self.torch_p2l = torch.FloatTensor(self.seq_p2l_lists)
        self.torch_pl2r = torch.FloatTensor(self.seq_pl2r_lists)
        self.torch_unt = torch.FloatTensor(self.seq_u_ntm_lists)
        self.torch_p2l_t = torch.LongTensor(self.seq_p2l_target_list)
        self.torch_pl2r_t = torch.LongTensor(self.seq_pl2r_target_list)
        self.torch_unt_t = torch.LongTensor(self.seq_u_ntm_target_list)

    def pretty_print_matrix(self, matrix):
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print '\n'.join(table)

    def is_digit(self, n):
        '''
        Test if n is a digit. Return true if so, false else.
        '''
        try:
            int(n)
            return True
        except ValueError:
            return False

    def print_rules_and_transfer_to_numpy(self):
        seq_input = self.seq_input_lists[0]
        self.seq_input_numpy = torch.LongTensor(len(seq_input))
        
        self.torch_lexicon = torch.LongTensor(self.num_term, self.num_non_term)
        
        self.torch_unary = torch.LongTensor(self.num_non_term, self.num_non_term)
        self.torch_binary = torch.LongTensor(self.num_non_term, self.num_non_term, self.num_non_term)

        #print "This is the lenghth of lexicon_numpy", len(self.lexicon_numpy)

        for i in xrange(len(seq_input)):
            self.seq_input_numpy[i] = seq_input[i]

        print "Lexicon is: "
        for i in xrange(self.num_term):
            for j in xrange(self.num_non_term):
                if self.lexicon[i][j]:
                    self.torch_lexicon[i,j] = 1
                    print "%s ---> %s" % (self.idx2Nonterm[j], self.idx2Word[i])                   

        print ""

        print "Unary nonterminal rules are: "
        for i in xrange(self.num_non_term):
            for j in xrange(self.num_non_term):
                if self.unary_dict[i][j]:
                    self.torch_unary[i,j] = 1
                    print "%s ---> %s" % (self.idx2Nonterm[j], self.idx2Nonterm[i])

        print ""

        print "Binary rules are: "
        for i in xrange(self.num_non_term):
            for j in xrange(self.num_non_term):
                for k in xrange(self.num_non_term):
                    if self.binary_dict[i][j][k]:
                        self.torch_binary[i, j, k] = 1
                        print "%s ---> %s %s" % (self.idx2Nonterm[k], self.idx2Nonterm[i], self.idx2Nonterm[j])

    def read_and_process(self):
        # TODO (@Vin) Change these filenames to be variables that can be passed in as
        # arguments. But for now, just leave it alone since it's easier to run the
        # model.

        # Preprocess the data, e.g. get parse trees and modify them to our need
        print "Reading and processing data ... "

        process_start = time.time()

        self.read_in_word_embeddings(self.dim_term)
        print "-Data Processor: Finish reading in word embeddings"
        self.create_nonterm_embeddings()
        print "-Data Processor: Finish reading in nonterminals feature mappings"

        # TODO (@Bo) Maybe we don't want to read in all inputs at once. We might want to
        # process them during training
        self.construct_seq_feats_lists()
        print "-Data Processor: Finish constructing list of matrices for inputs"
        # self.print_matrix(0)
        self.print_rules_and_transfer_to_numpy()

        process_end = time.time()
        process_time = process_end - process_start

        print "Finished, the reading and processing time is %.2f secs" % round(process_time, 0)

    def creat_log(self, log_dict):
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        print "creating training log file ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'w') as f:
            f.write('This the training log file ... \n')
            f.write('It tracks some statistics in the training process ... \n')
            #
            f.write('Model specs are listed below : \n')
            for the_key in log_dict['args']:
                f.write(
                    the_key+' : '+str(log_dict['args'][the_key])
                )
                f.write('\n')
            #
            f.write('Before training, the compilation time is '+str(log_dict['compile_time'])+' sec ... \n')
            f.write('Things that need to be tracked : \n')
            for the_key in log_dict['tracked']:
                f.write(the_key+' ')
            f.write('\n\n')
        #

    def continue_log(self, log_dict):
        print "continue tracking log ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'a') as f:
            for the_key in log_dict['tracked']:
                f.write(the_key+' is '+str(log_dict['tracked'][the_key])+' \n')
            f.write('\n')   

    def track_log(self, log_dict):
        print log_dict['iteration']
        print log_dict['tracked']['log_likelihood']    
