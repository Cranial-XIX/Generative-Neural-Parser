#!python
#cython: boundscheck=False
#cython: initializedcheck=False
###cython: wraparound=False     # TODO make sure there is no negative indexing first.
#cython: infertypes=True
#cython: cdivision=True
#distutils: language=c++

import os, sys
import time
import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log, exp
from numpy cimport int_t, double_t, int16_t

Vt = np.double
Dt = np.int16

ctypedef int_t      K_t
ctypedef int16_t    D_t
ctypedef double_t   V_t
ctypedef vector[short int] intvec

cdef packed struct Cell:
    V_t score
    D_t y
    D_t z
    D_t j
Cell_dt = np.dtype([('score', Vt), ('y', Dt), ('z', Dt), ('j', Dt)])

# left-child indexed binary rule
cdef packed struct BR:
    D_t right
    D_t parent
    V_t weight

# parent forward indexed binary rule
cdef packed struct BRF:
    D_t left
    D_t right
    V_t weight

# left-child indexed binary rule
cdef packed struct UR:
    D_t parent
    V_t weight

# new school
ctypedef vector[BR]    BRvv
ctypedef vector[UR]    URvv
ctypedef vector[BRF]   BRFvv
ctypedef BRvv    *BRv
ctypedef URvv    *URv
ctypedef BRFvv   *BRFv

cdef double log_zero = -100000
'''
cdef double logsumexp(double a, double b):
    cdef double m
    if a == log_zero:
        return b
    if b == log_zero:
        return a
    m = a if a > b else b
    return m + log(exp(a-m) + exp(b-m))
'''
cdef inline int tri(int i, int j) nogil:
    return j*(j-1)/2 + i

#TODODO rename GrammarObject to Parser
cdef class GrammarObject(object):

    cdef object nt2idx    # nonterminal to index
    cdef object idx2nt    # index to nonterminal
    cdef object w2idx     # word to index
    cdef object idx2w     # index to word
    cdef object prune_chart
    cdef object sen
    cdef object sentence

    cdef int num_nt, num_words, N

    cdef double log_zero

    cdef vector[BRv] rule_y_xz              # [NP] -> [(S,VP,log(0.5)), ...]
    cdef vector[BRFv] rule_x_yz
    cdef vector[URv] rule_y_x               # [S] -> [(ROOT,log(0.9)), ...]
    cdef vector[URv] lexicon                # [word] -> [(N,log(0.3)), ...]
    
    cdef vector[intvec*] spandex

    cdef double[:,:,:,:] betas
    cdef double[:,:,:,:] alphas
    cdef Cell[:,:,:] chart

    def __init__(self, processor=None):        
        if processor == None:  
            self.nt2idx = {}    # nonterminal to index
            self.idx2nt = []    # index to nonterminal
            self.w2idx = {}     # word to index
            self.idx2w = []     # index to word
        else:
            self.nt2idx = processor.nonterm2Idx
            self.idx2nt = processor.idx2Nonterm
            self.w2idx = processor.word2Idx
            self.idx2w = processor.idx2Word
            self.num_nt = processor.nt
            self.num_words = processor.nnt
       
       self._initialize()

    def check_files_exist(self, file_list):
        for file_name in file_list:
            if not os.path.exists(file_name):
                print 'Error! The file ', file_name, ' does not exist.'
                sys.exit()

    def _initialize(self):
        cdef:
            int nt, t

        for nt in xrange(self.num_nt):
            self.rule_y_x.push_back(new URvv())
            self.rule_y_xz.push_back(new BRvv())
            self.rule_x_yz.push_back(new BRFvv())
            self.rule_y_x_L.push_back(new URvv_L())
            self.rule_y_xz_L.push_back(new BRvv_L())
            self.rule_x_yz_L.push_back(new BRFvv_L())

        for t in xrange(self.num_words):
            self.lexicon.push_back(new URvv())
            
    def read_data_from_files(self, file_prefix, threshold=1e-7):
        nt_file = file_prefix + ".nonterminals"
        word_file = file_prefix + ".words"
        lex_file = file_prefix + ".lexicon"
        gr_file = file_prefix + ".grammar"

        self.check_files_exist([nt_file, word_file, lex_file, gr_file])

        self.read_nt_file(nt_file)
        self.read_word_file(word_file)
        self.read_lexicon_file(lex_file)
        self.read_gr_file(gr_file)
    
    def read_nt_file(self, nt_file):
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

    def read_word_file(self, word_file):
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

    def read_lexicon_file(self, lex_file):
        cdef UR ur
        
        # Read lexicon file   
        with open(lex_file, 'r') as file:
            for line in file:
                lexicon = line.strip().split()
                nt = self.nt2idx[lexicon[0]]
                if lexicon[1] in self.w2idx:
                    word = self.w2idx[lexicon[1]]
                else: # if word is OOV
                    word = 0
                ur.parent = nt
                ur.weight = float(lexicon[2].strip('[]'))
                if ur.weight >= threshold:
                    self.lexicon[word].push_back(ur)

    def read_gr_file(self, gr_file):
        cdef:
            UR ur
            BR br
            BRF brf

        # Read binary/unary rule file
        with open(gr_file, 'r') as file:
            for line in file:
                rule = line.strip().split()
                p = self.nt2idx[rule[0][:-2]]       # [:-2] is to remove "_0" from "NP_0" to form "NP"
                l = self.nt2idx[rule[2][:-2]]
                if len(rule) == 5:                  # binary rule
                    r = self.nt2idx[rule[3][:-2]]
                    br.right = r
                    br.parent = p
                    br.weight = float(rule[4])
                    self.rule_y_xz[l].push_back(br)
                    brf.left = l
                    brf.right = r
                    brf.weight = float(rule[4])
                    self.rule_x_yz[p].push_back(brf)
                if len(rule) == 4:                  # unary rule
                    if p != l:                      # Do not allow self-recurring X -> X rules
                        ur.parent = p
                        ur.weight = float(rule[3])
                        if ur.weight >= threshold:
                            self.rule_y_x[l].push_back(ur)

    def init_rule_probs(self, preterm, unt_pr, p2l_pr, pl2r_pr, threshold=1e-7):
        #TODODO initialize these matrices
        self.preterm = preterm
        self.unt_pr = unt_pr
        self.p2l_pr = p2l_pr
        self.pl2r_pr = pl2r_pr
        
    def __dealloc__(self):
        for x in self.spandex:
            del x

    cdef inline void inside_unary(self, int i, int k) nogil:
        cdef:
            int c, p
            intvec tmp
            intvec* cell
            UR ur
        cell = self.spandex[tri(i,k)]

        # free unary rule X->X
        for c in deref(cell):
            self.betas[i,k,c,1] += self.betas[i,k,c,0]

        for c in deref(cell):
            for ur in deref(self.rule_y_x[c]):
                p = ur.parent
                if self.betas[i,k,p,1] == 0:
                    tmp.push_back(p)
                self.betas[i,k,p,1] += self.betas[i,k,c,0] * ur.weight

        for c in tmp:
            cell.push_back(c)

    cpdef do_inside_outside(self, sentence):
        cdef:
            int i, j, k, tag, w, l, r, p, c, n, ri, pp, ik
            str word
            double parent, left, right, child
            double d, d_times_left, d_times_right
            UR ur
            BR br
            intvec tmp
            intvec* cell

        sen = []
        sentence = sentence.strip().split()

        n = len(sentence)
        self.N = n
        for i in xrange(n):
            word = sentence[i]
            sen.append(self.w2idx[word] if word in self.w2idx else 0)

        self.sen = sen
        ri = self.nt2idx['ROOT']

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # Do inside algorithm
        self.betas = np.zeros((n, n+1, self.num_nt, 2))
        t0 = time.time()
        # initialization
        for i in xrange(n):  # w-1 constituents
            w = sen[i]
            k = i+1
            cell = self.spandex[tri(i, k)]
            for ur in deref(self.lexicon[w]):
                tag = ur.parent
                if self.betas[i,k,tag,0] == 0:
                    cell.push_back(tag)
                self.betas[i,k,tag,0] = ur.weight
            # unary appending
            for c in deref(cell):
                self.betas[i,k,c,1] += self.betas[i,k,c,0]

            for c in deref(cell):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if self.betas[i,k,p,1] == 0:
                        tmp.push_back(p)
                    self.betas[i,k,p,1] += self.betas[i,k,c,0] * ur.weight

            for c in tmp:
                cell.push_back(c)
            tmp.clear()

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for l in deref(self.spandex[tri(i, j)]):
                        left = self.betas[i,j,l,1] 
                        for br in deref(self.rule_y_xz[l]):
                            r = br.right
                            right = self.betas[j,k,r,1]
                            if right == 0:
                                continue
                            p = br.parent
                            if self.betas[i,k,p,0] == 0:
                                cell.push_back(p)
                            self.betas[i,k,p,0] += br.weight * left * right
                # unary appending
                for c in deref(cell):
                    self.betas[i,k,c,1] += self.betas[i,k,c,0]

                for c in deref(cell):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if self.betas[i,k,p,1] == 0:
                            tmp.push_back(p)
                        self.betas[i,k,p,1] += self.betas[i,k,c,0] * ur.weight

                for c in tmp:
                    cell.push_back(c)
                tmp.clear()

        t1 = time.time()
        print "inside ",t1 - t0

        # Do outside algorithm
        self.alphas = np.zeros((n, n+1, self.num_nt, 2))
        self.alphas[0,n,ri,1] = 1.0

        for w in reversed(xrange(2, n+1)): # wide to narrow
            for i in xrange(n-w+1):
                k = i + w

                # unary
                for c in deref(self.spandex[tri(i, k)]):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if self.alphas[i,k,p,1] == 0:
                            continue
                        self.alphas[i,k,c,0] += self.alphas[i,k,p,1] * ur.weight
                for c in deref(self.spandex[tri(i, k)]):
                    self.alphas[i,k,c,0] += self.alphas[i,k,c,1]

                # binary
                for p in deref(self.spandex[tri(i, k)]):
                    parent = self.alphas[i,k,p,0]
                    if parent == 0:
                        continue
                    for j in xrange(i+1, k):
                        for brf in deref(self.rule_x_yz[p]):
                            l = brf.left
                            r = brf.right
                            left = self.betas[i,j,l,1]
                            if left == 0:
                                continue
                            right = self.betas[j,k,r,1]
                            if right == 0:
                                continue
                            d = parent * brf.weight
                            d_times_left = d * left
                            d_times_right = d * right
                            self.alphas[j,k,r,1] += d_times_left
                            self.alphas[i,j,l,1] += d_times_right

        for i in xrange(n):
            k = i+1
            for c in deref(self.spandex[tri(i, k)]):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if self.alphas[i,k,p,1] == 0:
                        continue
                    self.alphas[i,k,c,0] += self.alphas[i,k,p,1] * ur.weight
            for c in deref(self.spandex[tri(i, k)]):
                self.alphas[i,k,c,0] += self.alphas[i,k,c,1]

        print "outside ", time.time() - t1
        self.spandex.clear()

        return self.betas[0,n,ri,1]

    cpdef prune_the_chart(self, prob_sen, posterior_threshold):
        cdef:
            int i, j, nt, deleted=0
            double threshold, a, b

        threshold = prob_sen * posterior_threshold

        self.prune_chart = np.full((self.N, self.N+1, self.num_nt), False)
        for i in xrange(self.N):
            for j in xrange(i+1, self.N+1):
                for nt in xrange(self.num_nt):
                    a = self.alphas[i,j,nt,0]
                    b = self.betas[i,j,nt,1]
                    if b == 0 or a == 0:
                        continue
                    if a * b > threshold:
                        self.prune_chart[i,j,nt] = True
                    else:
                        deleted += 1
        print "Pruned ", deleted, " nonterminals"

    cpdef str parse(self, str sentence):
        cdef:
            int i, j, k, tag, w, l, r, p, c, n, ri, pp, ik
            str word
            double parent, left, right, child, newscore
            UR ur
            BR br
            intvec tmp
            intvec* cell

            Cell[:,:,:] chart

        n = self.N
        sen = self.sen
        self.sentence = sentence.strip().split()
        chart = np.zeros((n,n+1,self.num_nt), dtype=Cell_dt)

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        ri = self.nt2idx['ROOT']
        # Do inside algorithm
        t0 = time.time()

        # initialization
        for i in xrange(n):  # w-1 constituents
            w = sen[i]
            k = i+1
            cell = self.spandex[tri(i, k)]
            for ur in deref(self.lexicon[w]):
                tag = ur.parent
                if self.prune_chart[i,k,tag]:
                    if chart[i,k,tag].score == 0:
                        cell.push_back(tag)
                    chart[i,k,tag].score = ur.weight
                    chart[i,k,tag].y = -1
            # unary appending
            for c in deref(cell):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if self.prune_chart[i,k,p]:
                        newscore = chart[i,k,c].score * ur.weight
                        if newscore > chart[i,k,p].score:
                            if chart[i,k,p].score == 0:
                                tmp.push_back(p)
                            chart[i,k,p].score = newscore
                            chart[i,k,p].y = c
                            chart[i,k,p].z = -1
            for c in tmp:
                cell.push_back(c)
            tmp.clear()

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for l in deref(self.spandex[tri(i, j)]):
                        left = chart[i,j,l].score
                        for br in deref(self.rule_y_xz[l]):
                            r = br.right
                            right = chart[j,k,r].score
                            if right == 0:
                                continue
                            p = br.parent
                            if self.prune_chart[i,k,p]:
                                newscore = br.weight * left * right
                                if newscore > chart[i,k,p].score:
                                    if chart[i,k,p].score == 0:
                                        cell.push_back(p)
                                    chart[i,k,p].score = newscore
                                    chart[i,k,p].y = l
                                    chart[i,k,p].z = r
                                    chart[i,k,p].j = j
                # unary appending
                for c in deref(cell):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if self.prune_chart[i,k,p]:
                            newscore = chart[i,k,c].score * ur.weight
                            if newscore > chart[i,k,p].score:
                                if chart[i,k,p].score == 0:
                                    tmp.push_back(p) 
                                chart[i,k,p].score = newscore
                                chart[i,k,p].y = c
                                chart[i,k,p].z = -1
                for c in tmp:
                    cell.push_back(c)
                tmp.clear()
        t1 = time.time()
        self.spandex.clear()
        self.chart = chart
        print "parsing takes ",t1 - t0

        '''
        for i in xrange(n):
            for j in xrange(n+1):
                for nt in xrange(self.num_nt):
                    if chart[i,j,nt].score > 0:
                        print (i,j,self.idx2nt[nt])
        '''
        #print "root has proba", chart[0,n,ri]
        if self.chart[0,n,ri].score == 0:
            return ""   # No parse found
        return self.print_parse(0, n, ri)

    cpdef print_parse(self, int i, int k, int nt):
        cdef:
            int y, z, j
            double score

        y = self.chart[i,k,nt].y
        z = self.chart[i,k,nt].z
        score = self.chart[i,k,nt].score
        j = self.chart[i,k,nt].j

        if y == -1:
            # is terminal rule
            return "(" + self.idx2nt[nt] + " " + self.sentence[i] + ")"
        elif z == -1:
            # unary rule
            return  "(" + self.idx2nt[nt] + " "  \
                + self.print_parse(i, k, y) + ")" 
        else:
            # binary rule
            return  "(" + self.idx2nt[nt] + " " \
                + self.print_parse(i, j, y) + " " \
                + self.print_parse(j, k, z) + ")"
