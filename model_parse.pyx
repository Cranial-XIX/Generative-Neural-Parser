import itertools
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
cimport numpy as np


cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
#TODO cdef function ??
def parse(self, sen, output):
    cdef int i, length
    cdef int child, parent

    if self.cuda_flag:
        nll = Variable(torch.FloatTensor([0])).cuda()
    else:
        nll = Variable(torch.FloatTensor([0]))

    #TODO cdef sen as Variable or FloatTensor
    sen = sen.view(-1)
    left_context = self.coef_lstm * output[0]

    length = len(sen) - 1
    # every entry is a list of tuples, with each tuple indicate a potential nonterminal 
    # at this position (nonterminal idx, sum of log probability over the constituent)
    #TODO cdef inside
    inside = [[[] for i in xrange(length + 1)] for j in xrange(length + 1)]

    # a hashmap that stores the total prob of certain constituent
    cdef dict hash_map = {}

    ## Inside Algorithm
    root_idx = 2

    # Initialization

    # TODO(@Bo) speed up!
    tt0 = time.time()
    for i in xrange(length):
        child = sen.data[i+1]
        for parent in self.lexicon[child]:
            tm0 = time.time()
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
            tm1 = time.time()
            if self.verbose == 'yes':
                print "LEX (parent) ", parent, tm1 - tm0
    tt1 = time.time()
    if self.verbose == 'yes':
        print "LEXICON ", tt1-tt0, "---------------------------"

    # cdef child_tpl
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
    cdef int width, start, mid
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
        for x in hash_map:
            print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x][0]][1].data[0])
        return -1, None, None, -1, -1
    else:
        nll = -inside[0][length][ hash_map[tpl_map][0] ][1]
        # DEBUG
        #if self.verbose == 'yes':
        #    for x in hash_map:
        #        print "%d covers from %d to %d with prob %f" % (x[2], x[0], x[1], inside[x[0]][x[1]][hash_map[x][0]][1].data[0])
        return nll, inside, hash_map, length, root_idx