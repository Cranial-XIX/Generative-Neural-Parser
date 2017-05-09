#!python
#cython: boundscheck=False
#cython: initializedcheck=False
###cython: wraparound=False     # TODO make sure there is no negative indexing first.
#cython: infertypes=True
#cython: cdivision=True
#distutils: language=c++
#distutils: libraries=['stdc++']

import os, sys
import time
import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libc.math cimport log, exp
from numpy cimport int_t, double_t, int16_t

ctypedef int16_t    D_t
ctypedef double_t   V_t
ctypedef int_t      K_t

# left-child indexed binary rule
cdef packed struct BR:
    D_t right
    D_t parent
    V_t weight

# left-child indexed binary rule
cdef packed struct UR:
    D_t parent
    V_t weight

# new school
ctypedef vector[BR]    BRvv
ctypedef vector[UR]    URvv
ctypedef BRvv    *BRv
ctypedef URvv    *URv


def check_files_exist(file_list):
    for file_name in file_list:
        if not os.path.exists(file_name):
            print 'Error! The file ', file_name, ' does not exist.'
            sys.exit()

cdef class GrammarObject(object):

    cdef object nt2idx    # nonterminal to index
    cdef object idx2nt    # index to nonterminal
    cdef int num_nt
    cdef object w2idx     # word to index
    cdef object idx2w     # index to word
    cdef int num_words
    cdef double log_zero

    cdef object lexicon_dict                # (time) -> Set([NP, ...])
    cdef object binary_rules                # (S, NP, VP) -> log(0.5)

    cdef object binary_rule_forward_dict
    cdef object unary_rules                 # (ROOT, S) -> log(0.9)

    cdef object unary_rule_forward_dict     # (ROOT) -> [S, ...]

    cdef object sum_unary_combo             # (A,B) -> sum of {A -> B, A -> C -> B}
    cdef object max_unary_combo             # (A,B) -> max of {A -> B, A -> C -> B}
    cdef object C_in_max_unary_combo        # (A,B) -> C \in max of {A -> B, A -> C -> B}
    cdef object viterbi
    cdef object bp

    cdef vector[BRv] rule_y_xz              # [NP] -> [(S,VP,log(0.5)), ...]
    cdef vector[URv] rule_y_x               # [S] -> [(ROOT,log(0.9)), ...]
    cdef vector[URv] lexicon                # [word] -> [(N,log(0.3)), ...]

    cdef double[:,:,:] betas
    cdef double[:,:,:] alphas
    cdef double[:,:,:] prune_chart

    def __init__(self):
        """
		grammar
		"""

        self.nt2idx = {}    # nonterminal to index
        self.idx2nt = []    # index to nonterminal
        self.num_nt = 0
        self.w2idx = {}     # word to index
        self.idx2w = []     # index to word
        self.num_words = 0
        self.log_zero = -100000

        self.lexicon_dict = {}
        self.binary_rules = None
        self.binary_rule_forward_dict = {}

        self.unary_rules = None
        self.unary_rule_forward_dict = {}

        self.sum_unary_combo = None
        self.max_unary_combo = None
        self.C_in_max_unary_combo = None

    cdef double logsumexp(self, double a, double b):
        cdef double m
        if a == self.log_zero:
            return b
        if b == self.log_zero:
            return a
        m = a if a > b else b
        return m + log(exp(a-m) + exp(b-m))

    def _initialize(self):
        cdef:
            int nt, t

        for nt in xrange(self.num_nt):
            self.rule_y_x.push_back(new URvv())
            self.rule_y_xz.push_back(new BRvv())
            self.unary_rule_forward_dict[nt] = []
            self.binary_rule_forward_dict[nt] = []

        for t in xrange(self.num_words):
            self.lexicon.push_back(new URvv())

    def read_grammar(self, filename):
        cdef:
            UR ur
            BR br

        nt_file = filename + ".nonterminals"
        word_file = filename + ".words"
        lex_file = filename + ".lexicon"
        gr_file = filename + ".grammar"

        check_files_exist([nt_file, word_file, lex_file, gr_file])

        # Read nonterminal file
        with open(nt_file, 'r') as file:
            i = -2
            for line in file:
                if i < 0:  # skip the first two lines
                    i += 1
                    continue
                nt = line.strip().split()[0]
                self.nt2idx[nt] = i
                self.idx2nt.append(nt)
                i += 1
        self.num_nt = i

        # Read word file
        with open(word_file, 'r') as file:
            self.w2idx['OOV'] = 0
            self.idx2w.append('OOV')
            i = 1
            for line in file:
                w = line.strip()
                self.w2idx[w] = i
                self.idx2w.append(w)
                i += 1
        self.num_words = i

        self._initialize()

        # Read lexicon file        
        with open(lex_file, 'r') as file:
            for line in file:
                lexicon = line.strip().split()
                nt = self.nt2idx[lexicon[0]]
                if lexicon[1] in self.w2idx:
                    word = self.w2idx[lexicon[1]]
                else:  # if word is OOV
                    word = 0
                ur.parent = nt
                ur.weight = log(float(lexicon[2].strip('[]')))
                self.lexicon[word].push_back(ur)

        # Read binary/unary rule file
        self.binary_rules = [[[0 for k in xrange(self.num_nt)] for j in xrange(self.num_nt)] for i in xrange(self.num_nt)]
        with open(gr_file, 'r') as file:
            for line in file:
                rule = line.strip().split()
                parent = self.nt2idx[rule[0][:-2]]    # [:-2] is to remove "_0" from "NP_0" to form "NP"
                l = self.nt2idx[rule[2][:-2]]
                if len(rule) == 5:  # binary rule
                    r = self.nt2idx[rule[3][:-2]]
                    self.binary_rules[parent][l][r] = float(rule[4])
                    self.binary_rule_forward_dict[parent].append((l, r))
                    br.right = r
                    br.parent = parent
                    br.weight = log(float(rule[4]))
                    self.rule_y_xz[l].push_back(br)
                if len(rule) == 4:  # unary rule
                    if parent != l:    # Do not allow self-recurring X -> X rules
                        #TODO redundant
                        ur.parent = parent
                        ur.weight = log(float(rule[3]))
                        self.rule_y_x[l].push_back(ur)
                        self.unary_rule_forward_dict[parent].append(l)

    def compute_sum_and_max_of_unary_combos(self):
        cdef:
            int p, c
            UR ur

        self.sum_unary_combo = np.full((self.num_nt, self.num_nt), self.log_zero)
        self.max_unary_combo = np.full((self.num_nt, self.num_nt), self.log_zero)
        self.C_in_max_unary_combo = np.zeros((self.num_nt, self.num_nt))

        # p = parent, c = child
        for c in xrange(self.num_nt):
            for ur in deref(self.lexicon[c]):
                p = ur.parent
                weight = ur.weight
                if weight > self.log_zero:
                    self.sum_unary_combo[p, c] = self.logsumexp(self.sum_unary_combo[p, c], weight)
                    if weight > self.max_unary_combo[p, c]:
                        self.max_unary_combo[p, c] = weight
                        self.C_in_max_unary_combo[p, c] = -1

        # Handle sum and max unary combos, i.e. {A -> B, A -> C -> B}
        for c in xrange(self.num_nt):
            for ur in deref(self.rule_y_x[c]):
                p = ur.parent
                rule_prob = ur.weight  # C -> B
                if rule_prob == self.log_zero:
                    continue
                for ancestor_ur in deref(self.rule_y_x[p]):  # A
                    ancestor = ancestor_ur.parent
                    if ancestor_ur.weight > self.log_zero:
                        # prob of A -> C -> B
                        combo_rule_prob = ancestor_ur.weight + ur.weight
                #TODODO rm for ancestor in xrange(self.num_nt):         # A
                    # if self.unary_rules[ancestor][p] > self.log_zero:
                        # # prob of A -> C -> B
                        # combo_rule_prob = self.unary_rules[ancestor][p] + self.unary_rules[p][c]
                        self.sum_unary_combo[ancestor][c] = self.logsumexp(self.sum_unary_combo[ancestor][c], combo_rule_prob)
                        if combo_rule_prob > self.max_unary_combo[ancestor][c]:
                            self.max_unary_combo[ancestor][c] = combo_rule_prob
                            self.C_in_max_unary_combo[ancestor][c] = p


    def prune_unlikely_rules_and_lexicon(self, threshold):
        #TODO prune immediately when reading grammar file
    
        cdef int l, r
    
        # Prune lexicon
        for word in xrange(self.num_words):
            for ur in deref(self.lexicon[word]):
                if ur.weight < log(threshold):
                    ur.weight = self.log_zero

        # Prune binary rules
        for l in xrange(self.num_nt):
            for br in deref(self.rule_y_xz[l]):
                if br.weight < log(threshold):
                    self.binary_rules[br.parent][l][br.right] = self.log_zero

        # Prune unary rules
        pass #TODO dunno what to do yet
    
    
    def do_inside_outside(self, sentence):   
        words_in_sent = sentence.strip().split()
        cdef int n = len(words_in_sent)
        cdef int i, tag, w, j, l, r, p, c
        cdef double rule_prob, tag_prob
        cdef double[:,:,:] betas
        cdef UR ur

        cdef int ri = self.nt2idx['ROOT']

        t0 = time.time()
        # Do inside algorithm
        betas = np.full((n, n+1, self.num_nt), self.log_zero) #[[[0 for k in xrange(self.num_nt)] for j in xrange(n+1)] for i in xrange(n)]
        self.betas = betas
        # initialization
        for i in xrange(n):  # w-1 constituents
            if words_in_sent[i] in self.w2idx:
                word = words_in_sent[i]
            else:  # if word is OOV
                word = 'OOV'
            for ur in deref(self.lexicon[self.w2idx[word]]):
                tag = ur.parent
                tag_prob = ur.weight
                if tag_prob == self.log_zero:
                    continue
                betas[i,i+1,tag] = self.logsumexp(betas[i,i+1,tag], tag_prob)

                # Unary appending
                for ur in deref(self.rule_y_x[tag]):
                    p = ur.parent
                    betas[i,i+1,p] = self.logsumexp(betas[i,i+1,p], (self.sum_unary_combo[p][tag] + tag_prob))

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                for j in xrange(i+1, k):
                    for l in xrange(self.num_nt):
                        if betas[i,j,l] == self.log_zero:
                            continue
                        for br in deref(self.rule_y_xz[l]):
                            rule_prob = br.weight + betas[i,j,l] + betas[j,k,br.right]
                            if rule_prob > self.log_zero:
                                betas[i,k,br.parent] = self.logsumexp(betas[i,k,br.parent], rule_prob)

                # Unary appending
                for p in xrange(self.num_nt):
                    if betas[i,k,p] == self.log_zero:
                        continue
                    for ur in deref(self.rule_y_x[p]):
                        unary_p = ur.parent
                        betas[i,k,unary_p] = self.logsumexp(betas[i,k,unary_p], (self.sum_unary_combo[unary_p][p] + betas[i,k,p]))

        t1 = time.time()
        #print "inside takes ", t1 - t0

        # Do outside algorithm
        self.alphas = np.full((n, n+1, self.num_nt), self.log_zero)
        self.alphas[0,n,ri] = 0

        cdef double out

        for w in reversed(xrange(1, n+1)): # wide to narrow
            for i in xrange(n - w + 1):
                k = i + w
                for p in xrange(self.num_nt):
                    out_p = self.alphas[i,k,p]
                    if out_p == self.log_zero:
                        continue
                    # unary
                    for c in self.unary_rule_forward_dict[p]:
                        if betas[i,k,c] == self.log_zero:
                            continue
                        self.alphas[0,n,c] = self.logsumexp(self.alphas[0,n,c], (self.sum_unary_combo[p][c] + out_p))

                    if w == 1:
                        continue

                    # binary
                    for j in xrange(i + 1, k):
                        for (l, r) in self.binary_rule_forward_dict[p]:
                            if betas[i,j,l] == self.log_zero or betas[j,k,r] == self.log_zero:
                                continue
                            out = self.binary_rules[p][l][r] + out_p
                            # Skipping \alphas[A -> BC]
                            self.alphas[i,j,l] = self.logsumexp(self.alphas[i,j,l], (out + betas[j,k,r]))            
                            self.alphas[j,k,r] = self.logsumexp(self.alphas[j,k,r], (out + betas[i,j,l]))

        #print "outside takes ", time.time() - t1
        return betas[0,n,ri]

    def prune_the_chart(self, sentence, log_prob_sentence, posterior_threshold):
        cdef int n, i, j
    
        words_in_sent = sentence.strip().split()
        n = len(words_in_sent)
        if posterior_threshold == 0:
            log_posterior_threshold = float("-inf")
        else:
            log_posterior_threshold = log(posterior_threshold)
        log_unnormalized_threshold = log_posterior_threshold + log_prob_sentence

        # TODO use BooleanTensor instead of LongTensor
        self.prune_chart = np.zeros((n, n+1, self.num_nt))
        for i in xrange(n):
            for j in xrange(i+1, n+1):
                for nonterminal in xrange(self.num_nt):
                    if self.betas[i,j,nonterminal] == self.log_zero or self.alphas[i,j,nonterminal] == self.log_zero:
                        continue
                    #TODODO re if self.betas[i,j,nonterminal] + self.alphas[i,j,nonterminal] > log_unnormalized_threshold:
                    self.prune_chart[i,j,nonterminal] = 1

    cpdef str parse(self, str sentence):
        cdef int n, i, tag, w, j, l, r, p, c
    
        words_in_sent = sentence.strip().split()
        n = len(words_in_sent)
        #print "before aaaaa: ", betas[0][n][self.nt2idx['ROOT']]
        # Do inside algorithm
        self.viterbi = np.full((n,n+1,self.num_nt), self.log_zero)
        self.bp = [[[None for k in xrange(self.num_nt)] for j in xrange(n+1)] for i in xrange(n)]

        for i in xrange(n):  # w-1 constituents
            if words_in_sent[i] in self.w2idx:
                word = words_in_sent[i]
            else:  # if word is OOV
                #print 'Found OOV word: ', words_in_sent[i]
                word = 'OOV'
            for ur in deref(self.lexicon[self.w2idx[word]]):
                tag = ur.parent
                if not self.prune_chart[i,i+1,tag]:
                    continue 
                tag_prob = ur.weight
            # for tag in xrange(self.num_nt):
                # if not self.prune_chart[i,i+1,tag]:
                    # continue               
                # tag_prob = self.lexicon[tag][self.w2idx[word]]
                if tag_prob == self.log_zero:
                    continue
                self.viterbi[i][i+1][tag] = tag_prob

                # Unary appending 
                for ur in deref(self.rule_y_x[tag]):
                    p = ur.parent
                    if not self.prune_chart[i,i+1,p]:
                        continue
                    prob = self.max_unary_combo[p][tag] + tag_prob
                    if prob > self.viterbi[i][i+1][p]:
                        self.viterbi[i][i+1][p] = prob
                        c = self.C_in_max_unary_combo[p][tag]
                        if c == -1:
                            self.bp[i][i+1][p] = (None, None, tag)
                        else:
                            self.bp[i][i+1][p] = (None, None, c)

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n - w + 1):
                k = i + w
                for j in xrange(i + 1, k):
                    for l in xrange(self.num_nt):
                        if self.viterbi[i][j][l] == self.log_zero:
                            continue
                        for br in deref(self.rule_y_xz[l]):
                            if not self.prune_chart[i,k,br.parent]:
                                continue
                            rule_prob = self.binary_rules[br.parent][l][br.right] + self.viterbi[i][j][l] + self.viterbi[j][k][br.right]
                            if rule_prob > self.viterbi[i][k][br.parent]:
                                self.viterbi[i][k][br.parent] = rule_prob
                                self.bp[i][k][br.parent] = (j, l, br.right)

                # Unary appending
                for p in xrange(self.num_nt):
                    if self.viterbi[i][k][p] == self.log_zero:
                        continue
                    for ur in deref(self.rule_y_x[p]):
                        unary_p = ur.parent
                        if not self.prune_chart[i,k,unary_p]:
                            continue
                        u_prob = self.max_unary_combo[unary_p][p] + self.viterbi[i][k][p]
                        if u_prob > self.viterbi[i][k][unary_p]:
                            self.viterbi[i][k][unary_p] = u_prob
                            self.bp[i][k][unary_p] = (None, None, p)

        if not self.bp[0][n][self.nt2idx['ROOT']] == None:
            return self.print_parse(0, n, self.nt2idx['ROOT'], words_in_sent)
        else:
            return ""

    def print_parse(self, i, j, node, words_in_sent):
        next = self.bp[i][j][node]
        if next == None:
            # is terminal rule
            return "(" + self.idx2nt[node] + " " + words_in_sent[i] + ")"
        elif next[0] == None:
            # unary rule
            return  "(" + self.idx2nt[node] + " "  \
                + self.print_parse(i, j, next[2], words_in_sent) + ")" 
        else:
            # binary rule
            return  "(" + self.idx2nt[node] + " " \
                + self.print_parse(i, next[0], next[1], words_in_sent) + " " \
                + self.print_parse(next[0], j, next[2], words_in_sent) + ")"
            
    def validate_read_grammar(self):
        cdef int key, i, j, k
    
        for key in self.nt2idx:
            print key, self.nt2idx[key]

        for i in xrange(self.num_nt):
            print i, self.idx2nt[i]

        for key in self.w2idx:
            print key, self.w2idx[key]

        for i in xrange(self.num_words):
            print i, self.idx2w[i]

        for word in xrange(self.num_words):
            for ur in deref(self.lexicon[word]):          
                if ur.weight > self.log_zero:
                    print self.idx2nt[ur.parent], self.idx2w[word], ur.weight

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                for k in xrange(self.num_nt):
                    if self.binary_rules[i][j][k] > self.log_zero:
                        print self.idx2nt[i], self.idx2nt[j], self.idx2nt[k], self.binary_rules[i][j][k]

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.unary_rules[i][j] > self.log_zero:
                    print self.idx2nt[i], self.idx2nt[j], self.unary_rules[i][j]
                
        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.sum_unary_combo[i][j] > self.log_zero:
                    print self.idx2nt[i], self.idx2nt[j], self.sum_unary_combo[i][j]

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.max_unary_combo[i][j] > self.log_zero:
                    print self.idx2nt[i], self.idx2nt[j], self.max_unary_combo[i][j]

    def debinarize(self, parse):
        cdef int i
    
        if parse == None:
            return "NO_PARSE"
        stack = [1 for x in xrange(len(parse))]
        newparse = []
        pointer = -1
        flag = -1
        for i in xrange(len(parse)-1):
            if parse[i] == '(' and parse[i+1] == '@':
                pointer += 1
                flag = i+3
                continue
            if parse[i] == '(':
                stack[pointer] += 1
            if parse[i] == ')' and pointer >= 0:
                stack[pointer] -= 1
                if stack[pointer] == 0:
                    pointer -= 1
                    continue
            if flag == -1:
                newparse.append(parse[i])
            elif flag == i:
                flag = -1
        return "("+"".join(newparse[5:])

    #for i in xrange(len(grammer_obj.sum_unary_combo[0])):
    #    print "Pr Root -> ", grammer_obj.idx2nt[i], " is ", grammer_obj.sum_unary_combo[0][i]
