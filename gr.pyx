#!python
#cython: boundscheck=False
#cython: initializedcheck=False
###cython: wraparound=False     # TODO make sure there is no negative indexing first.
#cython: infertypes=True
#cython: cdivision=True
#distutils: language=c++

import os
import sys
import time
import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport log, exp
from libcpp.map cimport map
from libcpp.vector cimport vector

from numpy cimport int_t, double_t, int16_t

Vt = np.double
Dt = np.int16

ctypedef int_t      K_t
ctypedef int16_t    D_t
ctypedef double_t   V_t
ctypedef vector[short int] intvec

# cell in cky parsing chart
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

# parent forward indexed binary rule
cdef packed struct BRF:
    D_t left
    D_t right

# left-child indexed binary rule
cdef packed struct UR:
    D_t parent
    D_t idx

# new school
ctypedef vector[BR]    BRvv
ctypedef vector[UR]    URvv
ctypedef vector[BRF]   BRFvv
ctypedef BRvv    *BRv
ctypedef URvv    *URv
ctypedef BRFvv   *BRFv

cdef double LOG_ZERO = -1000000
cdef int RI = 2
cdef int U_NTM = 1
'''
cdef double logsumexp(double a, double b):
    cdef double m
    if a == LOG_ZERO:
        return b
    if b == LOG_ZERO:
        return a
    m = a if a > b else b
    return m + log(exp(a-m) + exp(b-m))
'''
cdef inline int tri(int i, int j) nogil:
    return j*(j-1)/2 + i

cdef inline int pl2rhash(int p, int l, int i, int j) nogil:
    return p*90900+l*900+i*30+j

#TODO(@Bo) rename GrammarObject to Parser
cdef class GrammarObject(object):

    cdef object nt2idx    # nonterminal to index
    cdef object idx2nt    # index to nonterminal
    cdef object w2idx     # word to index
    cdef object idx2w     # index to word
    cdef object prune_chart
    cdef object sen
    cdef object sentence
    cdef object unary_prefix
    cdef object unary_suffix

    cdef int num_nt, num_words, N

    cdef vector[BRv] rule_y_xz              # [NP] -> [(S,VP,log(0.5)), ...]
    cdef vector[BRFv] rule_x_yz
    cdef vector[URv] rule_y_x               # [S] -> [(ROOT,log(0.9)), ...]
    cdef vector[URv] lexicon                # [word] -> [(N,log(0.3)), ...]
    cdef vector[intvec*] spandex

    cdef map[int, int] pl2r_map
    cdef map[int, double] preterminal, unary, binary
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
            self.nt2idx = processor.nt2idx
            self.idx2nt = processor.idx2nt
            self.w2idx = processor.w2idx
            self.idx2w = processor.idx2w
            self.num_nt = processor.nnt
            self.num_words = processor.nt
            self.unary_prefix = processor.unary_prefix
            self.unary_suffix = processor.unary_suffix

        self._initialize(processor)

    def check_files_exist(self, file_list):
        for file_name in file_list:
            if not os.path.exists(file_name):
                print 'Error! The file ', file_name, ' does not exist.'
                sys.exit()

    def _initialize(self, processor):
        cdef:
            int nt, t, index
            UR ur

        for nt in xrange(self.num_nt):
            self.rule_y_x.push_back(new URvv())
            self.rule_y_xz.push_back(new BRvv())
            self.rule_x_yz.push_back(new BRFvv())

        for t in xrange(self.num_words):
            self.lexicon.push_back(new URvv())

        if not processor == None:
            index = -1
            for chain in processor.idx2u:
                index += 1
                bot = chain[0]
                top = chain[-1]
                ur.parent = top
                ur.idx = index
                self.rule_y_x[bot].push_back(ur)

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
                self.lexicon[word].push_back(ur)

    def read_gr_file(self, gr_file):
        cdef:
            BR br
            BRF brf

        # Read binary/unary rule file
        with open(gr_file, 'r') as file:
            for line in file:
                rule = line.strip().split()
                p = self.nt2idx[rule[0][:-2]]          # [:-2] is to remove "_0" from "NP_0" to form "NP"
                l = self.nt2idx[rule[2][:-2]]
                if len(rule) == 5:                     # binary rule
                    r = self.nt2idx[rule[3][:-2]]
                    br.right = r
                    br.parent = p
                    self.rule_y_xz[l].push_back(br)
                    brf.left = l
                    brf.right = r
                    self.rule_x_yz[p].push_back(brf)

    def __dealloc__(self):
        for x in self.spandex:
            del x

    cpdef viterbi(self,
        str sentence,
        np.ndarray[np.float32_t, ndim=2] preterm,
        np.ndarray[np.float32_t, ndim=3] unt,
        np.ndarray[np.float32_t, ndim=3] p2l,
        np.ndarray[np.float32_t, ndim=2] pl2r):

        cdef:
            int i, j, k, w, n, ik, index
            int tag, l, r, p, c
            double parent, left, right, child, score
            UR ur
            BR br
            intvec tmp
            intvec* cell
            Cell[:,:,:] chart

        parse_start = time.time()

        self.sentence = sentence.strip().split()
        n = self.N

        chart = np.empty((n,n+1,self.num_nt), dtype=Cell_dt)

        for i in xrange(n):
            for k in xrange(i+1,n+1):
                for p in xrange(self.num_nt):
                    chart[i,k,p].score = LOG_ZERO

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # initialize the chart
        for i in xrange(n):
            k = i + 1
            cell = self.spandex[tri(i, k)]
            for tag in xrange(self.num_nt):
                score = preterm[i,tag]
                if score == LOG_ZERO:
                    continue
                cell.push_back(tag)
                tmp.push_back(tag)
                chart[i,k,tag].score = score
                chart[i,k,tag].y = -1

            # unary appending
            for c in tmp:
                child = chart[i,k,c].score
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    index = ur.idx
                    score = child + p2l[p,i,U_NTM] + unt[p,i,index]
                    if score > chart[i,k,p].score:
                        if chart[i,k,p].score == LOG_ZERO:
                            cell.push_back(p)
                        chart[i,k,p].score = score
                        chart[i,k,p].y = c
                        chart[i,k,p].z = -index-1
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

                            if right <= LOG_ZERO:
                                continue

                            p = br.parent
                            score = left + right + p2l[p,i,l] + pl2r[self.pl2r_map[pl2rhash(p,l,i,j)],r]

                            if score > chart[i,k,p].score:
                                if chart[i,k,p].score == LOG_ZERO:
                                    cell.push_back(p)
                                    tmp.push_back(p)
                                chart[i,k,p].score = score
                                chart[i,k,p].y = l
                                chart[i,k,p].z = r
                                chart[i,k,p].j = j
                # unary appending
                for c in tmp:
                    child = chart[i,k,c].score
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        index = ur.idx
                        score = child + p2l[p,i,U_NTM] + unt[p,i,index]
                        if score > chart[i,k,p].score:
                            if chart[i,k,p].score == LOG_ZERO:
                                cell.push_back(p)
                            chart[i,k,p].score = score
                            chart[i,k,p].y = c
                            chart[i,k,p].z = -index-1
                tmp.clear()

        parse_end = time.time()

        self.spandex.clear()
        self.pl2r_map.clear()
        self.chart = chart

        print "parsing takes ", parse_end - parse_start

        if self.chart[0,n,RI].score == LOG_ZERO:
            return "( ROOT )"

        return self.print_parse(0, n, RI)

    cpdef mbr(self, 
        str sentence,
        np.ndarray[np.float32_t, ndim=2] preterm,
        np.ndarray[np.float32_t, ndim=3] unt,
        np.ndarray[np.float32_t, ndim=3] p2l,
        np.ndarray[np.float32_t, ndim=2] pl2r):

        cdef:
            int i, j, k, w, n, ik
            int tag, l, r, p, c
            double parent, left, right, child, score, BETA_ROOT
            UR ur
            BR br
            intvec tmp
            intvec* cell
            Cell[:,:,:] chart

        parse_start = time.time()
        self.sentence = sentence.strip().split()
        # Get alpha and beta values
        n = self.N

        BETA_ROOT = self.inside_outside(preterm, unt, p2l, pl2r)

        # Initialize chart
        chart = np.zeros((n,n+1,self.num_nt), dtype=Cell_dt)

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # Do inside algorithm

        # Initialization
        for i in xrange(n):  # w-1 constituents
            k = i + 1
            cell = self.spandex[tri(i, k)]
            # append preterminals
            for tag in xrange(self.num_nt):
                if preterm[i,tag] == 0:
                    continue
                cell.push_back(tag)
                tmp.push_back(tag)
                chart[i,k,tag].score = self.preterminal_edge(i,tag)
                chart[i,k,tag].y = -1

            # unary appending
            for c in tmp:
                child = chart[i,k,c].score
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    score = child + self.unary_edge(i,k,p,c)
                    if score > chart[i,k,p].score:
                        if chart[i,k,c].z == -1 and chart[i,k,c].y != -1:  # prev is unary
                            if (chart[i,k,chart[i,k,c].y].z == -1 and
                                chart[i,k,chart[i,k,c].y].y != -1):        # prev-prev is unary
                                continue
                        if chart[i,k,p].score == 0:
                            cell.push_back(p)
                        chart[i,k,p].score = score
                        chart[i,k,p].y = c
                        chart[i,k,p].z = -1
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
                            score = left + right + self.binary_edge(i,j,k,p,l,r)

                            if score > chart[i,k,p].score:
                                if chart[i,k,p].score == 0:
                                    cell.push_back(p)
                                    tmp.push_back(p)
                                chart[i,k,p].score = score
                                chart[i,k,p].y = l
                                chart[i,k,p].z = r
                                chart[i,k,p].j = j

                # unary appending
                for c in tmp:
                    child = chart[i,k,c].score
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        score = child + self.unary_edge(i,k,p,c)
                        if score > chart[i,k,p].score and chart[i,k,c].z != -1:
                            if chart[i,k,p].score == 0:
                                cell.push_back(p)
                            chart[i,k,p].score = score
                            chart[i,k,p].y = c
                            chart[i,k,p].z = -1
                tmp.clear()

        parse_end = time.time()

        self.spandex.clear()
        self.pl2r_map.clear()
        self.chart = chart

        print "parsing takes ", parse_end - parse_start

        if self.chart[0,n,RI].score == 0:
            return "( ROOT )"

        return self.print_parse(0, n, RI)

    cdef inline double preterminal_edge(self, int i, int p):
        return self.preterminal[self.pkey(i,p)]

    cdef inline double unary_edge(self, int i, int k, int p, int c):
        return self.unary[self.ukey(i,k,p,c)]

    cdef inline double binary_edge(self, int i, int j, int k, int p, int l, int r):
        return self.binary[self.bkey(i,j,k,p,l,r)]

    cpdef inside_outside(self,
                np.ndarray[np.float32_t, ndim=2] preterm,
                np.ndarray[np.float32_t, ndim=3] unt,
                np.ndarray[np.float32_t, ndim=3] p2l,
                np.ndarray[np.float32_t, ndim=2] pl2r):
        '''
        The inside outside using matrix probabilities
        '''

        cdef:
            int i, j, k, w, n, ik
            int tag, l, r, p, c
            double parent, left, right, child, score
            double d, d_times_left, d_times_right
            UR ur
            BR br
            BRF brf
            intvec tmp
            intvec* cell

        n = self.N
        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # Do inside algorithm
        self.betas = np.zeros((n, n+1, self.num_nt, 2))
        t0 = time.time()
        # initialization
        for i in xrange(n):  # w-1 constituents
            k = i + 1
            cell = self.spandex[tri(i, k)]
            for tag in xrange(self.num_nt):
                score = preterm[i,tag]
                if score == 0:
                    continue
                cell.push_back(tag)
                self.betas[i,k,tag,0] = score

            # unary appending
            for c in deref(cell):
                self.betas[i,k,c,1] += self.betas[i,k,c,0]

            for c in deref(cell):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if self.betas[i,k,p,1] == 0:
                        tmp.push_back(p)
                    self.betas[i,k,p,1] += \
                        self.betas[i,k,c,0] * p2l[p,i,U_NTM] * unt[p,i,c]
            for p in tmp:
                cell.push_back(p)
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
                            self.betas[i,k,p,0] += p2l[p,i,l] * pl2r[self.pl2r_map[pl2rhash(p,l,i,j)],r] * left * right
                # unary appending
                for c in deref(cell):
                    self.betas[i,k,c,1] += self.betas[i,k,c,0]

                for c in deref(cell):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if self.betas[i,k,p,1] == 0:
                            tmp.push_back(p)
                        self.betas[i,k,p,1] += self.betas[i,k,c,0] * p2l[p,i,U_NTM] * unt[p,i,c]
                for p in tmp:
                    cell.push_back(p)
                tmp.clear()

        t1 = time.time()
        print "inside ",t1 - t0

        # Do outside algorithm
        self.alphas = np.zeros((n, n+1, self.num_nt, 2))
        self.alphas[0,n,RI,1] = 1.0

        for w in reversed(xrange(2, n+1)): # wide to narrow
            for i in xrange(n-w+1):
                k = i + w

                # unary
                for c in deref(self.spandex[tri(i, k)]):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if self.alphas[i,k,p,1] == 0:
                            continue
                        d = self.alphas[i,k,p,1] * p2l[p,i,U_NTM] * unt[p,i,c]
                        self.alphas[i,k,c,0] += d
                        score = (d * self.betas[i,k,c,0])
                        if score > 0:
                            self.unary[self.ukey(i,k,p,c)] += score

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
                            d = parent * p2l[p,i,l] * pl2r[self.pl2r_map[pl2rhash(p,l,i,j)],r]
                            d_times_left = d * left
                            d_times_right = d * right
                            self.alphas[j,k,r,1] += d_times_left
                            self.alphas[i,j,l,1] += d_times_right
                            score = d_times_left * right
                            if score > 0:
                                self.binary[self.bkey(i,j,k,p,l,r)] += score

        for i in xrange(n):
            k = i+1
            for c in deref(self.spandex[tri(i, k)]):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if self.alphas[i,k,p,1] == 0:
                        continue
                    d = self.alphas[i,k,p,1] * p2l[p,i,U_NTM] * unt[p,i,c]
                    self.alphas[i,k,c,0] += d
                    score = (d * self.betas[i,k,c,0])
                    if score > 0:
                        self.unary[self.ukey(i,k,p,c)] += score

            for c in deref(self.spandex[tri(i, k)]):
                self.alphas[i,k,c,0] += self.alphas[i,k,c,1]

            for tag in xrange(self.num_nt):
                left = preterm[i, tag]
                if left == 0:
                    continue
                score = self.alphas[i,k,tag,0] * left
                if score > 0:
                    self.preterminal[self.pkey(i,tag)] += score

        print "outside ", time.time() - t1
        self.spandex.clear()

        return self.betas[0,n,RI,1]

    cdef inline int bkey(self, int i, int j, int k, int p, int l, int r) nogil:
        return (((((i*self.N) + j)*(self.N+1) + k)*self.num_nt + p)*self.num_nt + l)*self.num_nt + r

    cdef inline int ukey(self, int i, int k, int p, int c) nogil:
        return (((i*self.N) + k)*self.num_nt + p)*self.num_nt + c

    cdef inline int pkey(self, int i, int p) nogil:
        return i*self.num_nt + p

    cpdef preprocess(self, int n, np.ndarray[np.float32_t, ndim=2] preterm):
        cdef:
            int i, j, k, w, ik
            int tag, l, r, p, c
            UR ur
            BR br
            intvec* cell
            intvec tmp, tmp2, pl2r_p, pl2r_l, pl2r_pi, pl2r_ci

            np.ndarray[np.int8_t, ndim=4] betas
            np.ndarray[np.int64_t, ndim=1] np_p
            np.ndarray[np.int64_t, ndim=1] np_l
            np.ndarray[np.int64_t, ndim=1] np_pi
            np.ndarray[np.int64_t, ndim=1] np_ci

        self.N = n
        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # Do inside algorithm
        betas = np.zeros((n, n+1, self.num_nt, 2), dtype=np.int8)

        size = -1
        preprocess_start = time.time()
        # initialization
        for i in xrange(n):  # w-1 constituents
            k = i+1
            cell = self.spandex[tri(i, k)]
            for tag in xrange(self.num_nt):
                if preterm[i,tag] == 0:
                    continue
                if betas[i,k,tag,0] == 0:
                    cell.push_back(tag)
                betas[i,k,tag,0] = 1

            # unary appending
            for c in deref(cell):
                betas[i,k,c,1] = 1

            for c in deref(cell):
                for ur in deref(self.rule_y_x[c]):
                    p = ur.parent
                    if betas[i,k,p,1] == 0:
                        tmp.push_back(p)
                    betas[i,k,p,1] = 1
            for p in tmp:
                cell.push_back(p)
            tmp.clear()

        t1 = time.time()
        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for l in deref(self.spandex[tri(i, j)]):
                        for br in deref(self.rule_y_xz[l]):
                            r = br.right
                            right = betas[j,k,r,1]
                            if right == 0:
                                continue
                            p = br.parent
                            hsh = pl2rhash(p,l,i,j)
                            if self.pl2r_map.count(hsh) == 0:
                                size += 1
                                pl2r_p.push_back(p)
                                pl2r_l.push_back(l)
                                pl2r_pi.push_back(i)
                                pl2r_ci.push_back(j)
                                self.pl2r_map[hsh] = size
                            if betas[i,k,p,0] == 0:
                                cell.push_back(p)
                            betas[i,k,p,0] = 1
                # unary appending
                for c in deref(cell):
                    betas[i,k,c,1] = 1

                for c in deref(cell):
                    for ur in deref(self.rule_y_x[c]):
                        p = ur.parent
                        if betas[i,k,p,1] == 0:
                            tmp.push_back(p)
                        betas[i,k,p,1] = 1
                for p in tmp:
                    cell.push_back(p)
                tmp.clear()
        t2 = time.time()
        size += 1
        np_p = np.zeros((size,), dtype=int)
        np_l = np.zeros((size,), dtype=int)
        np_pi = np.zeros((size,), dtype=int)
        np_ci = np.zeros((size,), dtype=int)

        for i in xrange(size):
            np_p[i] = pl2r_p.at(i)
            np_l[i] = pl2r_l.at(i)
            np_pi[i] = pl2r_pi.at(i)
            np_ci[i] = pl2r_ci.at(i)

        self.spandex.clear()
        pl2r_p.clear()
        pl2r_l.clear()
        pl2r_pi.clear()
        pl2r_ci.clear()
        preprocess_end = time.time()

        print "preprocess takes ", preprocess_end - t2, t2 - t1, t1 -preprocess_start
        return np_p, np_l, np_pi, np_ci

    cpdef prune(self, prob_sen, posterior_threshold):
        cdef:
            int i, j, nt, deleted=0
            double threshold, a, b

        threshold = prob_sen * posterior_threshold

        self.allowed = np.full((self.N, self.N+1, self.num_nt), False)
        for i in xrange(self.N):
            for j in xrange(i+1, self.N+1):
                for nt in xrange(self.num_nt):
                    a = self.alphas[i,j,nt,0]
                    b = self.betas[i,j,nt,1]
                    if b == 0 or a == 0:
                        continue
                    if a * b > threshold:
                        self.allowed[i,j,nt] = True
                    else:
                        deleted += 1
        print "Pruned ", deleted, " nonterminals"

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
        elif z < 0:
            # unary rule
            return  self.unary_prefix[-z-1] + self.print_parse(i, k, y) + self.unary_suffix[-z-1] 
        else:
            # binary rule
            return  "(" + self.idx2nt[nt] + " " \
                + self.print_parse(i, j, y) + " " \
                + self.print_parse(j, k, z) + ")"
