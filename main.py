import argparse
import constants
import datetime
import itertools
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from evalb import evalb, evalb_unofficial, evalb_many
from model import BLN, LN, BSN, BS
from nltk import Tree
from processor import Processor, PBLN, PLN
from ptb import ptb
from torch.autograd import Variable
from util import unbinarize, oneline

argparser = argparse.ArgumentParser(description='Generative Neural Parser')

# Below are variables associated with files
# =========================================================================
argparser.add_argument(
    '--data', default=constants.DATA_FILE, help='Data foler path'
)

argparser.add_argument(
    '--pretrain', default="", help='Pretrained file path'
)

argparser.add_argument(
    '--make-train', default="no", help='Whether to make new training instances'
)

argparser.add_argument(
    '--read-data', default="no", help='Whether to read in and preprocess training data'
)

# Below are variables associated with model
# =========================================================================
argparser.add_argument(
    '--model', default="LN", help='Model: BLN, LN, BSN, BS'
)

argparser.add_argument(
    '--mode', default="spv_train", help='Mode: spv_train, uspv_train, test, parse'
)

argparser.add_argument(
    '--seed', default=200, help='Random seed'
)

argparser.add_argument(
    '--lstm-coef', default=1.0, help='LSTM coefficient'
)

argparser.add_argument(
    '--lstm-layer', default=3, help='# LSTM layer'
)

argparser.add_argument(
    '--lstm-dim', default=300, help='LSTM hidden dimension'
)

argparser.add_argument(
    '--dropout', default=0, help='LSTM dropout'
)

argparser.add_argument(
    '--l2-coef', default=0, help='l2 norm coefficient'
)

argparser.add_argument(
    '--verbose', default="no", help='use verbose mode'
)

# Below are variables associated with training
# =========================================================================
argparser.add_argument(
    '--epochs', default=200, help='# epochs to train'
)

argparser.add_argument(
    '--batch-size', default=100, help='# instances in a batch'
)

argparser.add_argument(
    '--learning-rate', default=0.01, help="learning rate"
)

argparser.add_argument(
    '--cuda', action='store_true', help='enable CUDA training'
)

args = argparser.parse_args()
# Create folder to save model and log files
file_save = ""
save_template = "TIME={}_MDL={}_EPCH={}_BSIZE={}_HD={}_LY={}"

if args.mode == 'spv_train' or args.mode == 'uspv_train':

    name = save_template.format(
        time.strftime("%d-%m-%Y-%H:%M:%S"),
        args.model,
        args.epochs,
        args.batch_size, 
        args.lstm_dim,
        args.lstm_layer
    )

    os.makedirs('./output/' + name + '/')
    file_save = os.path.abspath('./output/' + name + '/' + 'model')


## show values ##
if args.verbose == 'yes':
    template = "{0:30}{1:35}"
    print "="*80
    print "- Files:"
    print template.format(" train file :", args.data)
    print template.format(" pretrained file :", args.pretrain)
    print template.format(" whether to read new data :", args.read_data)
    print "- Model:"
    print template.format(" model :", args.model)
    print template.format(" model mode :", args.mode)
    print template.format(" LSTM coefficient :", args.lstm_coef)
    print template.format(" LSTM # of layer :", args.lstm_layer)
    print template.format(" LSTM dimension :", args.lstm_dim)
    print template.format(" LSTM dropout :", args.dropout)
    print template.format(" l2 coefficient :", args.l2_coef)
    print "- Train:"
    print template.format(" # of epochs is :", args.epochs)
    print template.format(" batch size is :", args.batch_size)
    print template.format(" learning rate is :", args.learning_rate)
    print "="*80


args.batch_size = int(args.batch_size)
args.epochs = int(args.epochs)
args.learning_rate = float(args.learning_rate)
args.lstm_coef = float(args.lstm_coef)
args.lstm_layer = int(args.lstm_layer)
args.lstm_dim = int(args.lstm_dim)
args.dropout = float(args.dropout)
args.l2_coef = float(args.l2_coef)
args.make_train = (args.make_train == 'yes')
args.read_data = (args.read_data == 'yes')
args.seed = int(args.seed)
args.verbose = (args.verbose == 'yes')

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

# let the processor read in data
# Model: BLN, LN, BSN, BS
if args.model == 'BLN':
    dp = PBLN(args.data, args.make_train, args.read_data, args.verbose, args.seed)

elif args.model == 'LN':
    dp = PLN(args.data, args.make_train, args.read_data, args.verbose, args.seed)

elif args.model == 'BSN' or args.model == 'BS':
    dp = Processor(args.data, args.make_train, args.read_data, args.verbose, args.seed)

else:
    print "Cannot recognize the model!"
    sys.exit()

dp.process_data()

# set batch size for unsupervised learning and parsing
if not (args.mode == 'spv_train'):
    args.batch_size = 1

# input arguments for model
inputs = {
    'verbose': args.verbose,
    'cuda': args.cuda,

    # model
    'lstm_coef': args.lstm_coef,
    'nlayers': args.lstm_layer,
    'dropout': args.dropout,
    'bsz': args.batch_size,
    'dhid': args.lstm_dim,

    'dp': dp,
}

if args.model == 'BLN':
    model = BLN(inputs)
elif args.model == 'LN':
    model = LN(inputs)
elif args.model == 'BSN':
    model = BSN(inputs)
elif args.model == 'BS':
    model = BS(inputs)

if args.cuda:
    model.cuda()

if not args.pretrain == "":
    if args.verbose == 'yes':
        print " - use pretrained model from ", args.pretrain
    pretrain = torch.load(args.pretrain)
    model.load_state_dict(pretrain['state_dict'])


def supervised():
    # get model paramters
    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters()
    )

    learning_rate = args.learning_rate
    # define the optimizer to use; currently use Adam
    optimizer = optim.Adam(
        parameters, lr=learning_rate, weight_decay=args.l2_coef
    )

    total = dp.trainset_length

    template = "Epoch {} [{}/{} ({:.1f}%)] Time {:.2f}"

    max_F1 = 0
    model.train()

    for epoch in range(1, args.epochs+1):
        dp.shuffle()
        idx = 0
        batch = 0
        tot_loss = 0
        
        if epoch > 1 and learning_rate > 1e-4:
            parameters = itertools.ifilter(
                lambda x: x.requires_grad, model.parameters()
            )
            learning_rate *= 0.5
            optimizer = optim.Adam(
                parameters, lr=learning_rate, weight_decay=args.l2_coef
            )
      
        while True:
            start = time.time()
            batch += 1
            optimizer.zero_grad()

            # get next training instances
            idx, next_bch = dp.next(idx, args.batch_size)

            # create PyTorch Variables
            if args.cuda:
                next_bch = [Variable(torch.LongTensor(x).cuda()) for x in next_bch]
            else:
                next_bch = [Variable(torch.LongTensor(x)) for x in next_bch]

            # compute loss
            loss = model('supervised', next_bch)
            tot_loss += loss.data[0]

            loss.backward()

            optimizer.step()
            end = time.time()
	    
            if idx == -1:
                if args.verbose:
                    print template.format(epoch, total, total, 100., round(end - start, 5))
                break
            else:
                if args.verbose:
                    print template.format(epoch, idx, total,
                        float(idx)/total * 100., round(end - start, 5))


        print " Epoch {} -- likelihood of trainset is: {:.4f}\n".format(epoch, tot_loss)

        if epoch % 5 == 0:
       	    model.eval()
            #F1_train = test("train")
            #F1 = test("test")
            #model.parse_end()
            print "NLL on test : ", eval_test_likelihood()
            model.train()

        #if F1 > max_F1:
        #    max_F1 = F1
        #    torch.save( {'state_dict': model.state_dict()}, file_save )

    if args.verbose:
        print "\nFinish supervised training"


def parse(sentence, pret=None, p2l=None):
    sen = dp.get_idx(sentence)

    if args.cuda:
        sen = sen.cuda()

    return model.parse(sentence, Variable(sen), pret, p2l)


def fpr(want_and_got, got, want):
    "Compute F-measure, precition, and recall from |want & got|, |got|, |want|."
    if want == 0 and got == 0:
        # Wanted empty set, got empty set.
        r = p = f = 1
    elif want == 0 or got == 0 or want_and_got == 0:
        # Assuming previous case failed, then we have zero f1.
        f = 0
        r = 1.0 if want == 0 else want_and_got * 1.0 / want
        p = 1.0 if got == 0 else want_and_got * 1.0 / got
    else:
        r = want_and_got * 1.0 / want
        p = want_and_got * 1.0 / got
        f = 2 * p * r / (p + r)
    return f,p,r


def get_gold_partials(sentence, gold):
    n = len(sentence)
    preterminal_allowed = np.zeros(n).astype(int)
    p2l_allowed = np.zeros((n, dp.nnt)).astype(int)
    traverse_gold(gold, 0,  preterminal_allowed, p2l_allowed)
    return preterminal_allowed, p2l_allowed


def traverse_gold(tree, idx, pret, p2l):
    label = tree.label() # the current nonterminal label
    A = dp.nt2idx[label]

    if tree.height() == 2:
        # is leaf
        return idx, idx+1, A, [0, A]

    else:
        nchild = 0
        # a binary rule A -> B C or unary rule A -> B
        for subtree in tree:

            if nchild == 0:
                ii, jj, B, B_unary = traverse_gold(subtree, idx, pret, p2l)
            else:
                jj, kk, C, C_unary = traverse_gold(subtree, jj, pret, p2l)

            nchild += 1

        if nchild == 1:
            # unary rule
            if not B_unary == None:
                B_unary.append(A)

            return ii, jj, A, B_unary

        else:
            # binary rule
            if not B_unary == None:
                pret[ii] = dp.idx2u.index(B_unary)

            if not C_unary == None:
                pret[jj] = dp.idx2u.index(C_unary)

            p2l[ii, A] = 1

            return ii, kk, A, None


def eval_official(dataset, test_data):
    start = time.time()
    NLL_sum = 0
    expect = []
    got = []

    '''
    got2 = []
    NLL_sum2 = 0
    '''

    num_sen = 0
    template = "[{}/{} ({:.1f}%)] P: {:.2f} R: {:.2f} F1: {:.2f} NLL: {:.2f}"

    for (sentence, gold) in test_data:

        num_sen += 1
        #pret, p2l = get_gold_partials(sentence, gold)
        nll, parse_string, nll2, parse_string2 = parse(sentence)
        NLL_sum += nll
        parse_tree = Tree.fromstring(parse_string)
        #parse_tree2 = Tree.fromstring(parse_string2)
        expect.append(unbinarize(gold))
        got.append(unbinarize(parse_tree))
#        got2.append(unbinarize(parse_tree2))


    F = evalb_many(expect, got)

    end = time.time()
    
    print " On {} # Sen: {} F1: {:.5f} NLL: {:.2f} TIME: {:.2f}".format(
        dataset, N, F, NLL_sum, end-start ) 

    return F


def evalb_unofficial_helper(gold, parse):
    tree = Tree.fromstring(parse)
    GW, G, W = evalb_unofficial(
        oneline(unbinarize(gold)),
        tree
    )
    return GW, G, W

def eval_unofficial(dataset, test_data):
    start = time.time()
    N = len(test_data)
    GW_sum = G_sum = W_sum = NLL_sum = 0
    #GW_sum2 = G_sum2 = W_sum2 = NLL_sum2 = 0

    num_sen = 0
    template = "[{}/{} ({:.1f}%)] F1: {:.4f} NLL: {:.4f}"

    for (sentence, gold) in test_data:
        num_sen += 1
        #pret, p2l = get_gold_partials(sentence, gold)
        
        nll, parse_string, nll2, parse_string2 = parse(sentence)

        NLL_sum += nll
        #NLL_sum2 += nll2

        GW, G, W = evalb_unofficial_helper(gold, parse_string)
        GW_sum += GW
        G_sum += G
        W_sum += W

        #GW2, G2, W2 = evalb_unofficial_helper(gold, parse_string2)
        #GW_sum2 += GW2
        #G_sum2 += G2
        #W_sum2 += W2

        if args.verbose:
            F, P, R = fpr(GW, G, W)
            #F2, P2, R2 = fpr(GW2, G2, W2)
            print template.format(num_sen, N, num_sen/float(N)*100, F, nll)


    F, _, _ = fpr(GW_sum, G_sum, W_sum)
    #F2, _, _ = fpr(GW_sum2, G_sum2, W_sum2)

    end = time.time()

    print " On {} # Sen: {} F1: {:.4f} NLL: {:.4f} TIME: {:.2f}".format(
        dataset, N, F, NLL_sum, end-start ) 
    #print " On {} # Sen: {} F1: {:.5f} NLL: {:.2f} TIME: {:.2f}".format(
    #    dataset, N, F2, NLL_sum, end-start ) 
    #print ""

    return F


def test(dataset):

    model.parse_setup()
    test_data = list(ptb(dataset, minlength=3, maxlength=constants.MAX_TEST_SEN_LENGTH, n=500))

    return eval_unofficial(dataset, test_data)

def eval_test_likelihood():
    idx = 0
    batch = 0
    tot_loss = 0
    while True:
        batch += 1

        # get next training instances
        idx, next_bch = dp.next(idx, args.batch_size, dataset='test')

        # create PyTorch Variables
        if args.cuda:
            next_bch = [Variable(torch.LongTensor(x).cuda()) for x in next_bch]
        else:
            next_bch = [Variable(torch.LongTensor(x)) for x in next_bch]

        # compute loss
        loss = model('supervised', next_bch)
        tot_loss += loss.data[0]
        if idx == -1:
            break

    return tot_loss

def check_spv():
    pass

## run the model
if args.mode == 'spv_train':
    supervised()

elif args.mode == 'uspv_train':
    unsupervised()

elif args.mode == 'test':
    test("test")

elif args.mode == 'parse':
    model.parse_setup()
    print "Enter sentence, one per line (press \"Enter\" to quit):"
    while True:
        sentence = raw_input()
        if sentence == "":
            break
        parse(sentence)

elif args.mode == 'check_spv':
    check_spv()

else:
    print "Unknown mode!"
