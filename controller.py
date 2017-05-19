import pickle
import time
import itertools
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

import constants
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
        'nlayers': cmd_inp['layer_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary,
        'unt_pre': p.unt_pre,
        'p2l_pre': p.p2l_pre,
        'pl2r_pre': p.pl2r_pre,
        'parser': None
    }

    model = LCNPModel(inputs, cmd_inp['cuda'], cmd_inp['verbose'])
    if cmd_inp['cuda']:
        model.cuda()

    if not cmd_inp['pretrain'] == None:
        if cmd_inp['verbose'] == 'yes':
            print " - use pretrained model from ", cmd_inp['pretrain']
        pretrain = torch.load(cmd_inp['pretrain'])
        model.load_state_dict(pretrain['state_dict'])

    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters, lr=cmd_inp['learning_rate'],
        weight_decay=cmd_inp['coef_l2'])

    try:
        for epoch in range(cmd_inp['max_epoch']):
            if cmd_inp['verbose'] == 'yes':
                print "\nTraining epoch %d =====================================" % epoch
            idx = 0
            batch = 0
            while not idx == -1:
                idx = p.next(idx, batch_size)
                if not idx == -1:
                    batch += 1
                    if cmd_inp['verbose'] == 'yes':
                        print "\nBatch %d -----------------------------------------" % batch
                    train_start = time.time()
                    optimizer.zero_grad()
                    p_array = [Variable(p.sens),
                        Variable(p.p2l), Variable(p.pl2r),
                        Variable(p.unt), Variable(p.ut),

                        Variable(p.p2l_t), Variable(p.pl2r_t),
                        Variable(p.unt_t), Variable(p.ut_t),

                        Variable(p.p2l_hi), Variable(p.pl2r_hi),
                        Variable(p.unt_hi), Variable(p.ut_hi)]
                    if cmd_inp['cuda']:
                        p_array = [x.cuda() for x in p_array]
                    loss = model(p_array[0],
                                p_array[1], p_array[2],
                                p_array[3], p_array[4],
                                
                                p_array[5], p_array[6],
                                p_array[7], p_array[8],
                                
                                p_array[9], p_array[10],
                                p_array[11], p_array[12])

                    t0 = time.time()
                    if cmd_inp['verbose'] == 'yes':
                        print " - NLL Loss: ", loss
                    loss.backward()
                    t1 = time.time()

                    optimizer.step()
                    train_end = time.time()
                    if cmd_inp['verbose'] == 'yes':
                        print " - Training one batch: " \
                            "forward: %.4f, backward: %.4f, optimize: %.4f secs" \
                            % (
                                round(t0 - train_start, 5), 
                                round(t1 - t0, 5),
                                round(train_end - t1, 5)
                            )
        torch.save({
                'state_dict': model.state_dict()
            }, cmd_inp['save'])
    except KeyboardInterrupt:
        print " - Exiting from training early"
        torch.save({
                'state_dict': model.state_dict()
            }, cmd_inp['save'])

    if cmd_inp['verbose'] == 'yes':
        print "Finish supervised training"

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
        'nlayers': cmd_inp['layer_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary,
        'unt_pre': p.unt_pre,
        'p2l_pre': p.p2l_pre,
        'pl2r_pre': p.pl2r_pre,
        'parser': None
    }

    model = LCNPModel(inputs, cmd_inp['cuda'], cmd_inp['verbose'])
    if cmd_inp['cuda']:
        model.cuda()

    if not cmd_inp['pretrain'] == None:
        if cmd_inp['verbose'] == 'yes':
            print " - use pretrained model from ", cmd_inp['pretrain']
        pretrain = torch.load(cmd_inp['pretrain'])
        model.load_state_dict(pretrain['state_dict'])

    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters, lr=cmd_inp['learning_rate'],
        weight_decay=cmd_inp['coef_l2'])

    try:
        for epoch in range(cmd_inp['max_epoch']):
            if cmd_inp['verbose'] == 'yes':
                print "\nTraining epoch %d =====================================" % epoch
            idx = 0
            batch = 0
            while not idx == -1:
                idx = p.next(idx)
                if not idx == -1:
                    batch += 1
                    if cmd_inp['verbose'] == 'yes':
                        print "\nSentence %d -----------------------------------------" % batch
                    train_start = time.time()
                    optimizer.zero_grad()
                    if cmd_inp['cuda']:
                        p_sen = Variable(p.sen).cuda()
                    else:
                        p_sen = Variable(p.sen)
                    loss = model(p_sen)
                    if loss.data[0] > 0:
                        t0 = time.time()
                        if cmd_inp['verbose'] == 'yes':
                            print " - NLL Loss: ", loss
                        loss.backward()
                        t1 = time.time()

                        optimizer.step()
                        train_end = time.time()
                        if cmd_inp['verbose'] == 'yes':
                            print " - Training one batch: forward: %.4f, backward: %.4f, "\
                                "optimize: %.4f secs" % (
                                round(t0 - train_start, 5),
                                round(t1 - t0, 5),
                                round(train_end - t1, 5) )
                    else:
                        if cmd_inp['verbose'] == 'yes':
                            print "No parse for sentence: ", p.get_sen(p.sen.view(-1))
        torch.save({
                'state_dict': model.state_dict()
            }, cmd_inp['save'])
    except KeyboardInterrupt:
        print(' - Exiting from training early')
        torch.save({
                'state_dict': model.state_dict()
            }, cmd_inp['save'])

    if cmd_inp['verbose'] == 'yes':
        print "Finish unsupervised training"

def parse_LCNP(p, parser, sentence, cmd_inp):

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
        'nlayers': cmd_inp['layer_lstm'],
        'bsz': batch_size,
        'dhid': cmd_inp['dim_model'],
        'initrange': 1,
        'lexicon': p.lexicon,
        'urules': p.unary,
        'brules': p.binary,
        'unt_pre': p.unt_pre,
        'p2l_pre': p.p2l_pre,
        'pl2r_pre': p.pl2r_pre,
        'parser': parser
    }

    model = LCNPModel(inputs, cmd_inp['cuda'], cmd_inp['verbose'])
    if cmd_inp['cuda']:
        model.cuda()
    if cmd_inp['pretrain'] == None:
        pretrain = torch.load(constants.PRE_TRAINED_FILE)
        model.load_state_dict(pretrain['state_dict'])
    else:
        pretrain = torch.load(cmd_inp['pretrain'])
        model.load_state_dict(pretrain['state_dict'])

    inp = p.get_idx(sentence)

    var_inp = Variable(inp)
    if cmd_inp['cuda']:
        var_inp = var_inp.cuda()
    return model.parse(sentence, var_inp)

def print_parse(p, sen, bp, start, end, node):
    next = bp[start][end][node]
    if next == None:
        # is terminal rule
        return "(" + p.idx2Nonterm[node] + " " + sen[start] + ")"
    elif next[0] == None:
        # unary rule
        return  "(" + p.idx2Nonterm[node] + " "  \
            + print_parse(p, sen, bp, start, end, next[2]) + ")" 
    else:
        # binary rule
        return  "(" + p.idx2Nonterm[parent] + " " \
            + print_parse(p, sen, bp, start, next[0], next[1]) + " " \
            + print_parse(p, sen, bp, next[0], end, next[2]) + ")"    
        


