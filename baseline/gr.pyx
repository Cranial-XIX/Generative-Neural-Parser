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

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
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

    cdef object lexicons                    # (NP, time) -> 0.3
    cdef object lexicon_dict                # (time) -> Set([NP, ...])
    cdef object binary_rules                # (S, NP, VP) -> 0.5
    cdef object DISGUISED_TOAST            # (NP, VP) -> [S, ...]
    cdef vector[BRv] rule_y_xz              # [NP] -> [(S,VP,0.6), ...]
    cdef object binary_rule_forward_dict
    cdef object unary_rules                 # (ROOT, S) -> 0.6
    cdef vector[URv] rule_y_x               # [S] -> [(ROOT,0.9), ...]
    cdef object unary_rule_forward_dict     # (ROOT) -> [S, ...]
    
    cdef object sum_unary_combo             # (A,B) -> sum of {A -> B, A -> C -> B}
    cdef object max_unary_combo             # (A,B) -> max of {A -> B, A -> C -> B}
    cdef object C_in_max_unary_combo        # (A,B) -> C \in max of {A -> B, A -> C -> B}
    cdef object viterbi
    cdef object bp

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

        self.lexicons = None                    # (NP, time) -> 0.3
        self.lexicon_dict = {}                  # (time) -> Set([NP, ...])
        self.binary_rules = None                # (S, NP, VP) -> 0.5
        self.binary_rule_forward_dict = {}

        self.unary_rules = None                 # (ROOT, S) -> 0.6
        self.unary_rule_forward_dict = {}       # (ROOT) -> [S, ...]
        
        self.sum_unary_combo = None             # (A,B) -> sum of {A -> B, A -> C -> B}
        self.max_unary_combo = None             # (A,B) -> max of {A -> B, A -> C -> B}
        self.C_in_max_unary_combo = None        # (A,B) -> C \in max of {A -> B, A -> C -> B}
        
    def read_grammar(self, filename):
        cdef int nonterminal
        cdef UR ur
        cdef BR br
    
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
        self.num_nt = len(self.idx2nt)

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
        self.num_words = len(self.idx2w)

        # Read lexicon file        
        self.lexicons = [[0 for x in xrange(self.num_words+1)] for y in xrange(self.num_nt)] # index 0 in 2nd dimension is OOV
        with open(lex_file, 'r') as file:  
            for line in file:
                lexicon = line.strip().split()
                nt = self.nt2idx[lexicon[0]]
                if lexicon[1] in self.w2idx:
                    word = self.w2idx[lexicon[1]]
                else:  # if word is OOV
                    word = 0
                self.lexicons[nt][word] += float(lexicon[2].strip('[]'))
                if word not in self.lexicon_dict:
                    self.lexicon_dict[word] = set()
                self.lexicon_dict[word].add(nt)

        # Read binary/unary rule file    
        self.binary_rules = [[[0 for k in xrange(self.num_nt)] for j in xrange(self.num_nt)] for i in xrange(self.num_nt)]
        self.unary_rules = [[0 for k in xrange(self.num_nt)] for j in xrange(self.num_nt)]
        for nonterminal in xrange(self.num_nt):
            # Must initialize early, or KeyError can occur
            self.rule_y_x.push_back(new URvv())
            self.rule_y_xz.push_back(new BRvv())
            self.unary_rule_forward_dict[nonterminal] = []
            self.binary_rule_forward_dict[nonterminal] = []
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
                    br.weight = float(rule[4])
                    self.rule_y_xz[l].push_back(br)
                    #TODO self.DISGUISED_TOAST[ (l, r) ].append(parent)
                if len(rule) == 4:  # unary rule
                    if parent != l:    # Do not allow self-recurring X -> X rules
                        #TODO redundant
                        self.unary_rules[parent][l] = float(rule[3])
                        ur.parent = parent
                        ur.weight = float(rule[3])
                        self.rule_y_x[l].push_back(ur)
                        self.unary_rule_forward_dict[parent].append(l)
 
    def compute_sum_and_max_of_unary_combos(self):
        cdef int p, c
    
        self.sum_unary_combo = [[0 for x in xrange(self.num_nt)] for y in xrange(self.num_nt)]
        self.max_unary_combo = [[0 for x in xrange(self.num_nt)] for y in xrange(self.num_nt)]
        self.C_in_max_unary_combo = [[0 for x in xrange(self.num_nt)] for y in xrange(self.num_nt)]

        # p = parent, c = child
        for p in xrange(self.num_nt):
            for c in xrange(self.num_nt):
                rule_prob = self.unary_rules[p][c]
                if rule_prob != 0:
                    self.sum_unary_combo[p][c] += rule_prob
                    self.max_unary_combo[p][c] = rule_prob
                    self.C_in_max_unary_combo[p][c] = -1                

        # Handle sum and max unary combos, i.e. {A -> B, A -> C -> B}
        for p in xrange(self.num_nt):
            for c in xrange(self.num_nt):
                rule_prob = self.unary_rules[p][c]  # C- > B
                if rule_prob == 0:
                    continue
                for ancestor in xrange(self.num_nt):         # A
                    if self.unary_rules[ancestor][p] > 0:
                        # prob of A -> C -> B
                        combo_rule_prob = self.unary_rules[ancestor][p] * self.unary_rules[p][c]
                        self.sum_unary_combo[ancestor][c] += combo_rule_prob
                        if combo_rule_prob > self.max_unary_combo[ancestor][c]:
                            self.max_unary_combo[ancestor][c] = combo_rule_prob
                            self.C_in_max_unary_combo[ancestor][c] = p


    def prune_unlikely_rules_and_lexicon(self, threshold):
        cdef int l, r
    
        # Prune lexicon
        for word in self.lexicon_dict:
            for tag in self.lexicon_dict[word]:
                if self.lexicons[tag][word] < threshold:
                    self.lexicons[tag][word] = 0

        # Prune binary rules
        for l in xrange(self.num_nt):
            for br in deref(self.rule_y_xz[l]):
                if br.weight < threshold:
                    self.binary_rules[br.parent][l][br.right] = 0
            #TODODO rm for r in xrange(self.num_nt):
                # if (l, r) not in self.DISGUISED_TOAST:
                    # continue
                # for p in self.DISGUISED_TOAST[ (l, r) ]:
                    # if self.binary_rules[p][l][r] < threshold:
                        # self.binary_rules[p][l][r] = 0

        # Prune unary rules
        pass #TODO dunno what to do yet
    
    
    def do_inside_outside(self, sentence):   
        words_in_sent = sentence.strip().split()
        cdef int n = len(words_in_sent)
        cdef int i, tag, w, j, l, r, p, c
        cdef double rule_prob, tag_prob
        cdef double[:,:,:] betas, alphas
        cdef UR ur

        cdef int ri = self.nt2idx['ROOT']

        t0 = time.time()
        # Do inside algorithm
        betas = np.zeros((n, n+1, self.num_nt)) #[[[0 for k in xrange(self.num_nt)] for j in xrange(n+1)] for i in xrange(n)]
        self.betas = betas
        # initialization
        for i in xrange(n):  # w-1 constituents
            for tag in xrange(self.num_nt):
                if words_in_sent[i] in self.w2idx:
                    word = words_in_sent[i]
                else:  # if word is OOV
                    word = 'OOV'
                tag_prob = self.lexicons[tag][self.w2idx[word]]
                if tag_prob == 0:
                    continue
                betas[i,i+1,tag] += tag_prob

                # Unary appending
                for ur in deref(self.rule_y_x[tag]):
                    p = ur.parent
                    betas[i,i+1,p] += self.sum_unary_combo[p][tag] * tag_prob

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                for j in xrange(i+1, k):
                    for l in xrange(self.num_nt):
                        if betas[i,j,l] == 0:
                            continue
                        for r in xrange(self.num_nt):
                            if betas[j,k,r] == 0:
                                continue
                            if (l, r) not in self.DISGUISED_TOAST:
                                continue
                            for p in self.DISGUISED_TOAST[ (l, r) ]:
                                rule_prob = self.binary_rules[p][l][r] * betas[i,j,l] * betas[j,k,r]
                                if rule_prob > 0:
                                    betas[i,k,p] += rule_prob

                # Unary appending
                for p in xrange(self.num_nt):
                    if betas[i,k,p] == 0:
                        continue
                    for ur in deref(self.rule_y_x[p]):
                        unary_p = ur.parent
                        betas[i,k,unary_p] += self.sum_unary_combo[unary_p][p] * betas[i,k,p]

        t1 = time.time()
        #print "inside takes ", t1 - t0

        # Do outside algorithm
        self.alphas = np.zeros((n, n+1, self.num_nt))
        self.alphas[0,n,ri] = 1

        cdef double out

        for w in reversed(xrange(1, n+1)): # wide to narrow
            for i in xrange(n - w + 1):
                k = i + w
                for p in xrange(self.num_nt):
                    out_p = self.alphas[i,k,p]
                    if out_p == 0:
                        continue
                    # unary
                    for c in self.unary_rule_forward_dict[p]:
                        if betas[i,k,c] == 0:
                            continue
                        self.alphas[0,n,c] += self.sum_unary_combo[p][c] * out_p

                    if w == 1:
                        continue

                    # binary
                    for j in xrange(i + 1, k):
                        for (l, r) in self.binary_rule_forward_dict[p]:
                            if betas[i,j,l] == 0 or betas[j,k,r] == 0:
                                continue
                            out = self.binary_rules[p][l][r] * out_p
                            # Skipping \alphas[A -> BC]
                            self.alphas[i,j,l] += out * betas[j,k,r]                
                            self.alphas[j,k,r] += out * betas[i,j,l]

        #print "outside takes ", time.time() - t1
        return betas[0,n,ri]

    def prune_the_chart(self, sentence, prob_sentence, posterior_threshold):
        cdef int n, i, j
    
        words_in_sent = sentence.strip().split()
        n = len(words_in_sent)
        unnormalized_threshold = posterior_threshold * prob_sentence

        # TODO use BooleanTensor instead of LongTensor
        self.prune_chart = np.zeros((n, n+1, self.num_nt))
        for i in xrange(n):
            for j in xrange(i+1, n+1):
                for nonterminal in xrange(self.num_nt):
                    if self.betas[i,j,nonterminal] == 0 or self.alphas[i,j,nonterminal] == 0:
                        continue
                    if self.betas[i,j,nonterminal] * self.alphas[i,j,nonterminal] > unnormalized_threshold:
                        self.prune_chart[i,j,nonterminal] = 1

    def parse(self, sentence):
        cdef int i, tag, w, j, l, r, p, c

        words_in_sent = sentence.strip().split()
        n = len(words_in_sent)
        #print "before aaaaa: ", betas[0][n][self.nt2idx['ROOT']]
        # Do inside algorithm
        self.viterbi = [[[0 for k in xrange(self.num_nt)] for j in xrange(n+1)] for i in xrange(n)]
        self.bp = [[[None for k in xrange(self.num_nt)] for j in xrange(n+1)] for i in xrange(n)]

        for i in xrange(n):  # w-1 constituents
            for tag in xrange(self.num_nt):
                if not self.prune_chart[i][i+1][tag]:
                    continue
                if words_in_sent[i] in self.w2idx:
                    word = words_in_sent[i]
                else:  # if word is OOV
                    #print 'Found OOV word: ', words_in_sent[i]
                    word = 'OOV'
                tag_prob = self.lexicons[tag][self.w2idx[word]]
                if tag_prob == 0:
                    continue
                self.viterbi[i][i+1][tag] = tag_prob

                # Unary appending 
                for ur in deref(self.rule_y_x[tag]):
                    p = ur.parent
                    if not self.prune_chart[i][i+1][p]:
                        continue
                    prob = self.max_unary_combo[p][tag] * tag_prob
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
                        if self.viterbi[i][j][l] == 0:
                            continue
                        for r in xrange(self.num_nt):
                            if self.viterbi[j][k][r] == 0:
                                continue
                            if (l, r) not in self.DISGUISED_TOAST:
                                continue
                            for p in self.DISGUISED_TOAST[ (l, r) ]:
                                if not self.prune_chart[i][k][p]:
                                    continue
                                rule_prob = self.binary_rules[p][l][r] * self.viterbi[i][j][l] * self.viterbi[j][k][r]
                                if rule_prob > self.viterbi[i][k][p]:
                                    self.viterbi[i][k][p] = rule_prob
                                    self.bp[i][k][p] = (j, l, r)

                # Unary appending
                for p in xrange(self.num_nt):
                    if self.viterbi[i][k][p] == 0:
                        continue
                    for ur in deref(self.rule_y_x[p]):
                        unary_p = ur.parent
                        if not self.prune_chart[i][k][unary_p]:
                            continue
                        u_prob = self.max_unary_combo[unary_p][p] * self.viterbi[i][k][p]
                        if u_prob > self.viterbi[i][k][unary_p]:
                            self.viterbi[i][k][unary_p] = u_prob
                            self.bp[i][k][unary_p] = (None, None, p)

        def print_parse(i, j, node):
            next = self.bp[i][j][node]
            if next == None:
                # is terminal rule
                return "(" + self.idx2nt[node] + " " + words_in_sent[i] + ")"
            elif next[0] == None:
                # unary rule
                return  "(" + self.idx2nt[node] + " "  \
                    + print_parse(i, j, next[2]) + ")" 
            else:
                # binary rule
                return  "(" + self.idx2nt[node] + " " \
                    + print_parse(i, next[0], next[1]) + " " \
                    + print_parse(next[0], j, next[2]) + ")"

        if not self.bp[0][n][self.nt2idx['ROOT']] == None:
            return print_parse(0, n, self.nt2idx['ROOT'])
        else:
            return None

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

        for i in xrange(self.num_nt):
            for j in xrange(self.num_words):
                if self.lexicons[i][j] != 0:
                    print self.idx2nt[i], self.idx2w[j], self.lexicons[i][j]

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                for k in xrange(self.num_nt):
                    if self.binary_rules[i][j][k] != 0:
                        print self.idx2nt[i], self.idx2nt[j], self.idx2nt[k], self.binary_rules[i][j][k]

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.unary_rules[i][j] != 0:
                    print self.idx2nt[i], self.idx2nt[j], self.unary_rules[i][j]
                
        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.sum_unary_combo[i][j] != 0:
                    print self.idx2nt[i], self.idx2nt[j], self.sum_unary_combo[i][j]

        for i in xrange(self.num_nt):
            for j in xrange(self.num_nt):
                if self.max_unary_combo[i][j] != 0:
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
    # grammer_obj.validate_read_grammar()
