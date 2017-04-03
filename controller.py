import pickle
import time
import itertools
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from torch.autograd import Variable

from model import LCNPModel

def spv_train_LCNP(p, cmd_inp):
    batch_size = cmd_inp['batch_size']

    inputs = {
        # terminals
        'term_emb': p.term_emb,
        'nt': p.nt,
        'dt': p.dt,

        # nonterminals
        'nt_emb': p.nonterm_emb,
        'nnt': p.nnt,
        'dnt': p.dnt,

        # model
        'coef_lstm': cmd_inp['coef_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'nlayers': 1,
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary
    }
    
    model = LCNPModel(inputs)
    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters, lr=cmd_inp['learning_rate'],
        weight_decay=cmd_inp['coef_l2'])

    for epoch in range(cmd_inp['max_epoch']):
        print "\nTraining epoch %d =====================================" % epoch
        idx = 0
        batch = 0
        while not idx == -1:
            idx = p.next(idx, batch_size)
            if not idx == -1:
                batch += 1
                print "\nBatch %d -----------------------------------------" % batch
                train_start = time.time()
                optimizer.zero_grad()
                loss = model(Variable(p.sens),
                    Variable(p.p2l), Variable(p.pl2r),
                    Variable(p.unt), Variable(p.ut),

                    Variable(p.p2l_t), Variable(p.pl2r_t),
                    Variable(p.unt_t), Variable(p.ut_t),

                    Variable(p.p2l_hi), Variable(p.pl2r_hi),
                    Variable(p.unt_hi), Variable(p.ut_hi))

                t0 = time.time()
                print " - NLL Loss: ", loss
                loss.backward()
                t1 = time.time()

                optimizer.step()
                train_end = time.time()
                print " - Training one batch: forward: %.4f, backward: %.4f, "\
                    "optimize: %.4f secs" % (
                        round(t0 - train_start, 5),
                        round(t1 - t0, 5),
                        round(train_end - t1, 5) )

    print "Finish training"

def uspv_train_LCNP(p, cmd_inp):

    batch_size = 1
    inputs = {
        # terminals
        'term_emb': p.term_emb,
        'nt': p.nt,
        'dt': p.dt,

        # nonterminals
        'nt_emb': p.nonterm_emb,
        'nnt': p.nnt,
        'dnt': p.dnt,

        # model
        'coef_lstm': cmd_inp['coef_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'nlayers': 1,
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary
    }
    
    model = LCNPModel(inputs)
    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters, lr=cmd_inp['learning_rate'],
        weight_decay=cmd_inp['coef_l2'])

    for epoch in range(cmd_inp['max_epoch']):
        print "\nTraining epoch %d =====================================" % epoch
        idx = 0
        batch = 0
        while not idx == -1:
            idx = p.next(idx)
            if not idx == -1:
                batch += 1
                print "\nSentence %d -----------------------------------------" % batch
                train_start = time.time()
                optimizer.zero_grad()
                loss = model(Variable(p.sen))
                if loss[0] > 0:
                    t0 = time.time()
                    print " - NLL Loss: ", loss
                    loss.backward()
                    t1 = time.time()

                    optimizer.step()
                    train_end = time.time()
                    print " - Training one batch: forward: %.4f, backward: %.4f, "\
                        "optimize: %.4f secs" % (
                            round(t0 - train_start, 5),
                            round(t1 - t0, 5),
                            round(train_end - t1, 5) )
                else:
                    print "No parse for sentence: ", p.get_sen(p.sen)

    print "Finish training"

def parse_LCNP(p, sen2parse, cmd_inp):

    batch_size = 1
    inputs = {
        # terminals
        'term_emb': p.term_emb,
        'nt': p.nt,
        'dt': p.dt,

        # nonterminals
        'nt_emb': p.nonterm_emb,
        'nnt': p.nnt,
        'dnt': p.dnt,

        # model
        'coef_lstm': cmd_inp['coef_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'nlayers': 1,
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary
    }
    
    model = LCNPModel(inputs)   

    inp = p.get_idx(sen2parse)

    nll, chart, hashmap, end, idx = lcnp.parse(inp)

    if nll > 0: # exist parse
        print "The best parse negative log likelihood is ", nll
        print print_parse(p, chart, hashmap, 0, end, idx)
    else:
        print "No parses for the sentence."


def print_parse(p, cky_chart, hash_map, start, end, idx):
    tpl_map = (start, end, idx)
    parent, curr_log_prob, left_sib, child, mid = cky_chart[start][end][hash_map[tpl_map][0]]

    if left_sib == -2:
        # is terminal rule
        return "(" + p.idx2Nonterm[parent] + " " + p.idx2Word[child] + ")"
    elif left_sib >= 0:
        return  "(" + p.idx2Nonterm[parent] + " " + print_parse(p, cky_chart, hash_map, start, mid, left_sib) + " " \
        + print_parse(p, cky_chart, hash_map, mid, end, child) + ")"        
    else:
        return  "(" + p.idx2Nonterm[parent] + " "  + print_parse(p, cky_chart, hash_map, mid, end, child) + ")"      
        


