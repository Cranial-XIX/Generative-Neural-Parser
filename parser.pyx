#!python
#distutils: language=c++
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: cdivision=True

import os
import sys
import time
import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport log, exp
from libcpp.map cimport map
from libcpp.vector cimport vector

from numpy cimport double_t, int16_t

Vt = np.double
Dt = np.int16

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

# unary rule
cdef packed struct UR:
    D_t parent
    D_t idx

# preterminal rule
cdef packed struct PR:
    D_t parent
    D_t preterminal
    D_t idx

ctypedef vector[BR]    BRvv
ctypedef vector[UR]    URvv
ctypedef vector[BRF]   BRFvv
ctypedef vector[PR]    PRvv
ctypedef BRvv    *BRv
ctypedef URvv    *URv
ctypedef BRFvv   *BRFv
ctypedef PRvv    *PRv

cdef double LOG_ZERO = -1000000
cdef int ROOT = 1
cdef int MAX_SEN_LEN = 30

cdef inline int tri(int i, int j) nogil:
    return j*(j-1)/2 + i

cdef inline int x2y_h(int i, int x) nogil:
    return x*31+i

cdef inline int xy2z_h(int i, int j, int x, int y) nogil:
    return x*96100+y*961+i*31+j

cdef inline int x2u_h(int i, int x) nogil:
    return x*31+i

cdef inline int lex_h(int i, int x) nogil:
    return x*31+i

# Parser
cdef class Parser(object):

    cdef object sen
    cdef object sentence

    cdef object prefix
    cdef object suffix

    cdef object idx2nt

    cdef int nnt, nt, N  # number of nonterminals & terminals, sentence length

    cdef vector[BRv]  rule_y_xz
    cdef vector[BRFv] rule_x_yz
    cdef vector[URv]  rule_u_x
    cdef vector[PRv]  lexicon # word to preterminal unary chain

    # hashmap for corresponding rule tuples
    cdef map[int, int] x2y_m
    cdef map[int, int] xy2z_m
    cdef map[int, int] lex_m
    cdef map[int, int] x2u_m

    # chart to store inside and outside scores
    #cdef double[:,:,:,:] betas
    #cdef double[:,:,:,:] alphas

    # CKY cell chart that stores possible nonterminals
    cdef vector[intvec*] spandex

    # CKY chart that stores back pointers and scores
    cdef Cell[:,:,:] chart

    def __init__(self, p):        
        cdef:
            int nt, t, U, B, A, C, PT
            UR ur
            BR br
            BRF brf
            PR pr

        self.nnt = p.nnt
        self.nt = p.nt
        self.prefix = p.unary_prefix
        self.suffix = p.unary_suffix
        self.idx2nt = p.idx2nt

        ## intialize the grammar rules
        for nt in xrange(self.nnt):
            self.rule_u_x.push_back(new URvv())
            self.rule_y_xz.push_back(new BRvv())
            self.rule_x_yz.push_back(new BRFvv())

        for t in xrange(self.nt):
            self.lexicon.push_back(new PRvv())

        for B in p.unary:
            for U, A in p.unary[B]:
                ur.idx = U
                ur.parent = A
                self.rule_u_x[B].push_back(ur)

        for B in p.B_AC:
            for A, C in p.B_AC[B]:
                br.right = C
                br.parent = A
                self.rule_y_xz[B].push_back(br)

                brf.left = B
                brf.right = C
                self.rule_x_yz[A].push_back(brf)

        for t in p.w_U:
            for U, A, PT in p.w_U[t]:
                pr.parent = A
                pr.preterminal = PT
                pr.idx = U
                self.lexicon[t].push_back(pr)


    def __dealloc__(self):
        for x in self.spandex:
            del x
        for b in self.rule_y_xz:
            del b
        for u in self.rule_u_x:
            del u
        for br in self.rule_x_yz:
            del br
        for l in self.lexicon:
            del l

    cpdef preparse(self, sentence, sen):

        cdef:
            int n, ik, i, j, k, w
            int t, A, B, C, U, PT
            int hsh
            int lex_i, x2y_i, xy2z_i, x2u_i
            #double start, mid, end

            UR ur
            BR br
            PR pr

            intvec* cell
            intvec tmp, P_P, P_i, U_A, U_i, B_A, B_i, C_A, C_B, C_i, C_j

            np.ndarray[np.int8_t, ndim=4] betas

            np.ndarray[np.int64_t, ndim=1] py_P_P
            np.ndarray[np.int64_t, ndim=1] py_P_i
            np.ndarray[np.int64_t, ndim=1] py_U_A
            np.ndarray[np.int64_t, ndim=1] py_U_i
            np.ndarray[np.int64_t, ndim=1] py_B_A
            np.ndarray[np.int64_t, ndim=1] py_B_i
            np.ndarray[np.int64_t, ndim=1] py_C_A
            np.ndarray[np.int64_t, ndim=1] py_C_B
            np.ndarray[np.int64_t, ndim=1] py_C_i
            np.ndarray[np.int64_t, ndim=1] py_C_j

        #start = time.time()

        self.sen = sen[1:]
        self.sentence = sentence.split()
        n = self.N = len(self.sentence)

        self.x2y_m.clear()
        self.xy2z_m.clear()
        self.lex_m.clear()
        self.x2u_m.clear()

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # Do inside algorithm
        betas = np.zeros((n, n+1, self.nnt, 2), dtype=np.int8)

        lex_i = x2y_i = xy2z_i = x2u_i = -1

        # initialization
        for i in xrange(n):
            k = i + 1
            t = self.sen[i]
            cell = self.spandex[tri(i, k)]
            for pr in deref(self.lexicon[t]):

                PT = pr.preterminal
                A = pr.parent

                hsh = lex_h(i, PT)
                if self.lex_m.count(hsh) == 0:
                    lex_i += 1
                    self.lex_m[hsh] = lex_i
                    P_P.push_back(A)
                    P_i.push_back(i)

                hsh = x2u_h(i, A)
                if self.x2u_m.count(hsh) == 0:
                    x2u_i += 1
                    self.x2u_m[hsh] = x2u_i
                    U_A.push_back(A)
                    U_i.push_back(i)
                
                cell.push_back(A)
                betas[i,k,A,1] = 1

        for w in xrange(2, n+1):
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for B in deref(self.spandex[tri(i, j)]):
                        for br in deref(self.rule_y_xz[B]):
                            C = br.right
                            if betas[j,k,C,1] == 0:
                                continue
                            A = br.parent

                            hsh = xy2z_h(i, j, A, B)
                            if self.xy2z_m.count(hsh) == 0:
                                xy2z_i += 1
                                self.xy2z_m[hsh] = xy2z_i
                                C_A.push_back(A)
                                C_B.push_back(B)
                                C_i.push_back(i)
                                C_j.push_back(j)

                            if betas[i,k,A,0] > 0:
                                continue

                            hsh = x2y_h(i, A)

                            x2y_i += 1
                            self.x2y_m[hsh] = x2y_i
                            B_A.push_back(A)
                            B_i.push_back(i)
                            
                            cell.push_back(A)
                            betas[i,k,A,0] = 1
                            betas[i,k,A,1] = 1


                for B in deref(cell):
                    for ur in deref(self.rule_u_x[B]):
                        A = ur.parent
                        if betas[i,k,A,1] > 0:
                            continue
                        tmp.push_back(A)
                        hsh = x2u_h(i, A)
                        if self.x2u_m.count(hsh) == 0:
                            x2u_i += 1
                            self.x2u_m[hsh] = x2u_i
                            U_A.push_back(A)
                            U_i.push_back(i)       
                        betas[i,k,A,1] = 1

                for p in tmp:
                    cell.push_back(p)

                tmp.clear()

        mid = time.time()

        lex_i += 1
        x2u_i += 1
        x2y_i += 1
        xy2z_i += 1

        py_P_P = np.zeros((lex_i,), dtype=int)
        py_P_i = np.zeros((lex_i,), dtype=int)
        py_U_A = np.zeros((x2u_i,), dtype=int)
        py_U_i = np.zeros((x2u_i,), dtype=int)
        py_B_A = np.zeros((x2y_i,), dtype=int)
        py_B_i = np.zeros((x2y_i,), dtype=int)
        py_C_A = np.zeros((xy2z_i,), dtype=int)
        py_C_B = np.zeros((xy2z_i,), dtype=int)
        py_C_i = np.zeros((xy2z_i,), dtype=int)
        py_C_j = np.zeros((xy2z_i,), dtype=int)

        for i in xrange(lex_i):
            py_P_P[i] = P_P.at(i)
            py_P_i[i] = P_i.at(i)

        for i in xrange(x2u_i):
            py_U_A[i] = U_A.at(i)
            py_U_i[i] = U_i.at(i)

        for i in xrange(x2y_i):
            py_B_A[i] = B_A.at(i)
            py_B_i[i] = B_i.at(i)

        for i in xrange(xy2z_i):
            py_C_A[i] = C_A.at(i)
            py_C_B[i] = C_B.at(i)
            py_C_i[i] = C_i.at(i)
            py_C_j[i] = C_j.at(i)

        P_P.clear()
        P_i.clear() 
        U_A.clear() 
        U_i.clear()
        B_A.clear()
        B_i.clear()
        C_A.clear()
        C_B.clear()
        C_i.clear()
        C_j.clear()

        self.spandex.clear()

        #end = time.time()

        #print " -- Preparse takes {:.2f}, transform to numpy takes {:.2f}".format(end-mid, mid-start)

        return py_P_P, py_P_i, py_U_A, py_U_i, py_B_A, py_B_i, py_C_A, py_C_B, py_C_i, py_C_j

    cpdef viterbi(self,
        np.ndarray[np.float32_t, ndim=2] x2y,
        np.ndarray[np.float32_t, ndim=2] xy2z,
        np.ndarray[np.float32_t, ndim=2] x2u,
        np.ndarray[np.float32_t, ndim=2] lex):

        cdef:
            int ik, i, j, k, w, n
            int t, A, B, C, PT, U
            double parent, left, right, child, score
            UR ur
            BR br
            PR pr
            intvec tmp
            intvec* cell
            Cell[:,:,:] chart

        start = time.time()
        n = self.N

        chart = np.empty((n,n+1,self.nnt), dtype=Cell_dt)

        for i in xrange(n):
            for k in xrange(i+1,n+1):
                for A in xrange(self.nnt):
                    chart[i,k,A].score = LOG_ZERO

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # initialize the chart
        for i in xrange(n):
            k = i + 1
            t = self.sen[i]
            cell = self.spandex[tri(i, k)]
            for pr in deref(self.lexicon[t]):

                PT = pr.preterminal
                A = pr.parent
                U = pr.idx

                score = x2u[self.x2u_m[x2u_h(i,A)],U] + lex[self.lex_m[lex_h(i,PT)],t]

                if score > chart[i,k,A].score:
                    cell.push_back(A)
                    chart[i,k,A].score = score
                    chart[i,k,A].y = -1
                    chart[i,k,A].z = U

        #for i in xrange(self.nnt):
        #    if not chart[0,1,i].score == LOG_ZERO:
        #        print "V", i, chart[0,1,i].score, chart[0,1,i].z

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for B in deref(self.spandex[tri(i, j)]):
                        left = chart[i,j,B].score
                        for br in deref(self.rule_y_xz[B]):
                            C = br.right
                            right = chart[j,k,C].score

                            if right <= LOG_ZERO:
                                continue

                            A = br.parent

                            score = left + right + x2y[self.x2y_m[x2y_h(i,A)],B] \
                                + xy2z[self.xy2z_m[xy2z_h(i,j,A,B)],C]

                            if score > chart[i,k,A].score:
                                cell.push_back(A)
                                chart[i,k,A].score = score
                                chart[i,k,A].y = B
                                chart[i,k,A].z = C
                                chart[i,k,A].j = j


                for B in deref(cell):
                    child = chart[i,k,B].score
                    for ur in deref(self.rule_u_x[B]):
                        A = ur.parent
                        U = ur.idx

                        score = child + x2u[self.x2u_m[x2u_h(i,A)],U]

                        if score > chart[i,k,A].score:
                            tmp.push_back(A)
                            chart[i,k,A].score = score
                            chart[i,k,A].y = B
                            chart[i,k,A].z = U
                            chart[i,k,A].j = -1

                for p in tmp:
                    cell.push_back(p)

                tmp.clear()

        self.spandex.clear()
        self.chart = chart

        score = self.chart[0,n,ROOT].score

        if score == LOG_ZERO:
            return score, "( ROOT )"
        else:
            return score, self.print_parse(0, n, ROOT)


    cpdef viterbi_gld(self,
        np.ndarray[np.float32_t, ndim=2] x2y,
        np.ndarray[np.float32_t, ndim=2] xy2z,
        np.ndarray[np.float32_t, ndim=2] x2u,
        np.ndarray[np.float32_t, ndim=2] lex,
        np.ndarray[np.int64_t, ndim=1] pret,
        np.ndarray[np.int64_t, ndim=2] p2l):

        cdef:
            int ik, i, j, k, w, n
            int t, A, B, C, PT, U
            double parent, left, right, child, score
            UR ur
            BR br
            PR pr
            intvec tmp
            intvec* cell
            Cell[:,:,:] chart

        start = time.time()
        n = self.N

        chart = np.empty((n,n+1,self.nnt), dtype=Cell_dt)

        for i in xrange(n):
            for k in xrange(i+1,n+1):
                for A in xrange(self.nnt):
                    chart[i,k,A].score = LOG_ZERO

        for ik in xrange(n*(n+1)//2):
            self.spandex.push_back(new intvec())

        # initialize the chart
        for i in xrange(n):
            k = i + 1
            t = self.sen[i]
            cell = self.spandex[tri(i, k)]
            for pr in deref(self.lexicon[t]):

                PT = pr.preterminal
                A = pr.parent
                U = pr.idx
                if not pret[i] == U:
                    continue
                score = x2u[self.x2u_m[x2u_h(i,A)],U] + lex[self.lex_m[lex_h(i,PT)],t]

                if score > chart[i,k,A].score:
                    cell.push_back(A)
                    chart[i,k,A].score = score
                    chart[i,k,A].y = -1
                    chart[i,k,A].z = U

        #for i in xrange(self.nnt):
        #    if not chart[0,1,i].score == LOG_ZERO:
        #        print "V_gld", i, chart[0,1,i].score, chart[0,1,i].z

        for w in xrange(2, n+1):  # wider constituents
            for i in xrange(n-w+1):
                k = i + w
                cell = self.spandex[tri(i, k)]
                for j in xrange(i+1, k):
                    for B in deref(self.spandex[tri(i, j)]):
                        left = chart[i,j,B].score
                        for br in deref(self.rule_y_xz[B]):
                            C = br.right
                            right = chart[j,k,C].score

                            if right <= LOG_ZERO:
                                continue

                            A = br.parent
                            #if not p2l[i, A] == 1:
                            #    continue

                            score = left + right + x2y[self.x2y_m[x2y_h(i,A)],B] \
                                + xy2z[self.xy2z_m[xy2z_h(i,j,A,B)],C]

                            if score > chart[i,k,A].score:
                                cell.push_back(A)
                                chart[i,k,A].score = score
                                chart[i,k,A].y = B
                                chart[i,k,A].z = C
                                chart[i,k,A].j = j


                for B in deref(cell):
                    child = chart[i,k,B].score
                    for ur in deref(self.rule_u_x[B]):
                        A = ur.parent
                        U = ur.idx

                        score = child + x2u[self.x2u_m[x2u_h(i,A)],U]

                        if score > chart[i,k,A].score:
                            tmp.push_back(A)
                            chart[i,k,A].score = score
                            chart[i,k,A].y = B
                            chart[i,k,A].z = U
                            chart[i,k,A].j = -1

                for p in tmp:
                    cell.push_back(p)

                tmp.clear()

        self.spandex.clear()
        self.chart = chart

        score = self.chart[0,n,ROOT].score

        if score == LOG_ZERO:
            return score, "( ROOT )"
        else:
            return score, self.print_parse(0, n, ROOT)


    cpdef print_parse(self, int i, int k, int A):
        cdef:
            int B, C, j

        B = self.chart[i,k,A].y
        C = self.chart[i,k,A].z
        j = self.chart[i,k,A].j
        score = self.chart[i,k,A].score

        if B == -1:
            # is terminal rule
            #print " t: ", i, score
            return self.prefix[C] + " " + self.sentence[i] + self.suffix[C]
        elif j == -1:
            # unary rule
            #print " u: ", i, k, score
            return self.prefix[C] + self.print_parse(i, k, B) + self.suffix[C]
        else:
            # binary rule
            #print " b: ", i, j, k, self.idx2nt[A], "=>", self.idx2nt[B], self.idx2nt[C], score
            return  "(" + self.idx2nt[A] + " " \
                + self.print_parse(i, j, B) + " " \
                + self.print_parse(j, k, C) + ")" 
