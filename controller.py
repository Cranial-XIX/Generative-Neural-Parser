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

def batchify(data, bsz):
    size = data.size()
    bsz = int(bsz)
    nbatch = int(size[0] // bsz)
    data = data.narrow(0, 0, nbatch * bsz)
    if len(size) == 4: # list of 3D tensors
        data = data.view(nbatch, bsz, size[1], size[2], -1).contiguous()
    elif len(size) == 3: # list of 2D tensors
        data = data.view(nbatch, bsz, size[1], -1).contiguous()
    elif len(size) == 2: # list of 1D tensors
        data = data.view(nbatch, bsz, -1).contiguous()
    else:
        print "the input is not in correct form"

    #data = data.cuda()
    return data

def get_batch(i, inp, pre, p2l, p2l_t, pl2r, pl2r_t, unt, unt_t):
    inp0 = Variable(inp[i])
    pre0 = Variable(pre[i])
    p2l0 = Variable(p2l[i])
    pl2r0 = Variable(pl2r[i])
    unt0 = Variable(unt[i])
    p2lt = Variable(p2l_t[i])
    pl2rt = Variable(pl2r_t[i])
    untt = Variable(unt_t[i])
    return inp0, pre0, p2l0, p2lt, pl2r0, pl2rt, unt0, untt 

def get_batch_input(i, inp):
    return Variable(inp[i])

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def spv_train_LCNP(processor, input_model):
    '''
    this function is called to train the left context neural parser model
    '''

    batch_size = input_model['batch_size']
    processor.data()
    

    inp = batchify(processor.torch_inp, batch_size)
    pre = batchify(processor.torch_pre, batch_size)
    p2l = batchify(processor.torch_p2l, batch_size)
    pl2r = batchify(processor.torch_pl2r, batch_size)
    unt = batchify(processor.torch_unt, batch_size)
    p2l_t = batchify(processor.torch_p2l_t, batch_size)
    pl2r_t = batchify(processor.torch_pl2r_t, batch_size)
    unt_t = batchify(processor.torch_unt_t, batch_size)

    nbatch = len(inp)

    inputs = {
        # terminals
        'term_emb': processor.emb_term,
        'nt': processor.num_term,
        'dt': processor.dim_term,
        # nonterminals
        'nt_emb': processor.emb_non_term,
        'nnt': processor.num_non_term,
        'dnt': processor.dim_non_term,
        # model
        'coef_lstm': input_model['coef_lstm'],
        'bsz': 2,
        'dhid': processor.dim_model,
        'nlayers': 1,
        'initrange': 1,
        'lexicon': processor.torch_lexicon,
        'urules': processor.torch_unary,
        'brules': processor.torch_binary
    }
    
    lcnp = LCNPModel(inputs)
    parameters = itertools.ifilter(lambda p: p.requires_grad == True, lcnp.parameters())
    optimizer = optim.Adam(parameters, lr = input_model['learning_rate'], weight_decay=input_model['coef_l2'])
    print "There are %d sentences to train" % processor.num_sen
    for epi in range(input_model['max_epoch']):

        print "Training epoch %d =====================================" % epi

        for i in range(nbatch):
            train_start = time.time()
            inp0, pre0, p2l0, p2lt, pl2r0, pl2rt, unt0, untt = get_batch(i, inp, pre, p2l, p2l_t, pl2r, pl2r_t, unt, unt_t)
            optimizer.zero_grad()
            loss = lcnp(inp0, pre0, p2l0, p2lt, pl2r0, pl2rt, unt0, untt)
            t0 = time.time()
            print "The loss is ", loss
            loss.backward()
            t1 = time.time()
            optimizer.step()
            train_end = time.time()
            print "Training one instance needs %.4f, %.4f, %.4f secs" % (round(t0 - train_start, 5),round(t1 - t0, 5),round(train_end - t1, 5) )

    print "Finish training"


def uspv_train_LCNP(processor, input_model):

    batch_size = input_model['batch_size']
    processor.data()

    inp = batchify(processor.torch_inp, batch_size)

    nbatch = len(inp)

    inputs = {
        # terminals
        'term_emb': processor.emb_term,
        'nt': processor.num_term,
        'dt': processor.dim_term,
        # nonterminals
        'nt_emb': processor.emb_non_term,
        'nnt': processor.num_non_term,
        'dnt': processor.dim_non_term,
        # model
        'coef_lstm': input_model['coef_lstm'],
        'bsz': 2,
        'dhid': processor.dim_model,
        'nlayers': 1,
        'initrange': 1,
        'lexicon': processor.torch_lexicon,
        'urules': processor.torch_unary,
        'brules': processor.torch_binary
    }
    
    lcnp = LCNPModel(inputs)
    optimizer = optim.Adam(lcnp.l2, lr = input_model['learning_rate'], weight_decay=input_model['coef_l2'])
    print "There are %d sentences to train" % processor.num_sen
    for epi in range(input_model['max_epoch']):

        print "Training epoch %d =====================================" % epi

        for i in range(nbatch):
            train_start = time.time()
            optimizer.zero_grad()
            loss = lcnp(inp0, pre0, p2l0, p2lt, pl2r0, pl2rt, unt0, untt)
            t0 = time.time()
            print "The loss is ", loss
            loss.backward()
            t1 = time.time()
            optimizer.step()
            train_end = time.time()
            print "Training one instance needs %.4f, %.4f, %.4f secs" % (round(t0 - train_start, 5),round(t1 - t0, 5),round(train_end - t1, 5) )

            '''
            inp0 = get_batch_input(i, inp)
            lcnp.zero_grad()
            loss = lcnp(inp0)
            print "The Negative Log Likelihood is ", loss
            loss.backward()
            clipped_lr = lr * clip_gradient(lcnp, 0.5)
            for p in lcnp.parameters():
                p.data.add_(-clipped_lr, p.grad.data)
            train_end = time.time()

            print "Training one instance needs %.2f secs" % round(train_end - train_start, 5)
            '''
    print "Finish training"

def parse_LCNP(processor, pretrained_file_name, input_parse, input_model):
    if pretrained_file_name == "":
        print "Need some pre-defined log-linear weights and bias to parse !!!"
        return
    batch_size = 1
    processor.data()

    inp = batchify(processor.torch_inp, batch_size)

    nbatch = len(inp)

    inputs = {
        # terminals
        'term_emb': processor.emb_term,
        'nt': processor.num_term,
        'dt': processor.dim_term,
        # nonterminals
        'nt_emb': processor.emb_non_term,
        'nnt': processor.num_non_term,
        'dnt': processor.dim_non_term,
        # model
        'coef_lstm': input_model['coef_lstm'],
        'bsz': 1,
        'dhid': processor.dim_model,
        'coef_l2': input_model['coef_l2'],
        'nlayers': 1,
        'initrange': 1,
        'lexicon': processor.torch_lexicon,
        'urules': processor.torch_unary,
        'brules': processor.torch_binary
    }
    lcnp = LCNPModel(inputs)    

    inp0 = get_batch_input(0, inp)

    nll, chart, hashmap, end, idx = lcnp.parse(inp0)

    if nll > 0: # exist parse
        print "The best parse negative log likelihood is ", nll
        print print_parse(processor, chart, hashmap, 0, end, idx)


def print_parse(processor, cky_chart, hash_map, start, end, idx):
    tpl_map = (start, end, idx)
    parent, curr_log_prob, left_sib, child, mid = cky_chart[start][end][hash_map[tpl_map][0]]

    if left_sib == -2:
        # is terminal rule
        return "(" + processor.idx2Nonterm[parent] + " " + processor.idx2Word[child] + ")"
    elif left_sib >= 0:
        return  "(" + processor.idx2Nonterm[parent] + " " + print_parse(processor, cky_chart, hash_map, start, mid, left_sib) + " " \
        + print_parse(processor, cky_chart, hash_map, mid, end, child) + ")"        
    else:
        return  "(" + processor.idx2Nonterm[parent] + " "  + print_parse(processor, cky_chart, hash_map, mid, end, child) + ")"      
        


