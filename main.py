import argparse
import constants
import datetime
import evalb
import itertools
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from data_processor import Processor
from gr import GrammarObject
from model import LCNPModel
from nltk import Tree
from ptb import ptb
from torch.autograd import Variable
from util import unbinarize, oneline

argparser = argparse.ArgumentParser(description='Generative Neural Parser')

# Below are variables associated with files
# =========================================================================
argparser.add_argument(
    '--train', default=constants.TRAIN_FILE, help='Train file path'
)

argparser.add_argument(
    '--pretrain', default="", help='Pretrained file path'
)

argparser.add_argument(
    '--make-train', default="no", help='Whether to make a new train file'
)

argparser.add_argument(
    '--read-data', default="no", help='Whether read data'
)

# Below are variables associated with model
# =========================================================================
argparser.add_argument(
    '--mode', default="parse", help='mode: spv_train, uspv_train, test, parse'
)

argparser.add_argument(
    '--seed', default=12345, help='random seed'
)

argparser.add_argument(
    '--lstm-coef', default=1.0, help='LSTM coefficient'
)

argparser.add_argument(
    '--lstm-layer', default=3, help='# LSTM layer'
)

argparser.add_argument(
    '--lstm-dim', default=100, help='LSTM hidden dimension'
)

argparser.add_argument(
    '--l2-coef', default=1e-2, help='l2 norm coefficient'
)

argparser.add_argument(
    '--verbose', default="yes", help='Use verbose mode'
)

# Below are variables associated with training
# =========================================================================
argparser.add_argument(
    '--epochs', default=10, help='# epochs to train'
)

argparser.add_argument(
    '--batch-size', default=2, help='# instances in a batch'
)

argparser.add_argument(
    '--learning-rate', default=0.02, help="learning rate"
)

argparser.add_argument(
    '--cuda', action='store_true', help='enable CUDA training'
)

args = argparser.parse_args()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Create folder to save model and log files
file_save = ""
if not args.mode == 'parse':
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    name = 'PID='+str(id_process)+'_TIME='+time_current
    os.makedirs('./output/' + name + '/')
    file_save = os.path.abspath('./output/' + name + '/' + 'model_dict')

## show values ##
if args.verbose == 'yes':
    template = "{0:30}{1:20}"
    print "="*80
    print "- Files:"
    print template.format(" train file :", args.train)
    print template.format(" pretrained file :", args.pretrain)
    print template.format(" whether to read new data :", args.read_data)
    print "- Model:"
    print template.format(" model mode :", args.mode)
    print template.format(" seed is :", args.seed)
    print template.format(" LSTM coefficient :", args.lstm_coef)
    print template.format(" LSTM # of layer :", args.lstm_layer)
    print template.format(" LSTM dimension :", args.lstm_dim)
    print template.format(" l2 coefficient :", args.l2_coef)
    print "- Train:"
    print template.format(" # of epochs is :", args.epochs)
    print template.format(" batch size is :", args.batch_size)
    print template.format(" learning rate is :", args.learning_rate)
    print "="*80

args.seed = int(args.seed)
args.lstm_coef = float(args.lstm_coef)
args.lstm_layer = int(args.lstm_layer)
args.lstm_dim = int(args.lstm_dim)
args.l2_coef = float(args.l2_coef)
args.epochs = int(args.epochs)
args.batch_size = int(args.batch_size)
args.verbose = (args.verbose == 'yes')
args.read_data = (args.read_data == 'yes')
args.make_train = (args.make_train == 'yes')

# let the processor read in data
p = Processor(args.train, args.make_train, args.read_data, args.verbose)
p.process_data()

# create a grammar object
parser = GrammarObject(p)
parser.read_gr_file(constants.GR_FILE)

# set batch size for unsupervised learning and parsing
if not args.mode == 'spv_train':
    args.batch_size = 1

# input arguments for model
inputs = {
    'verbose': args.verbose,
    'cuda': args.cuda,

    # terminals
    'term_emb': p.term_emb,
    'nt': p.nt,
    'dt': p.dt,

    # nonterminals
    'nt_emb': p.nonterm_emb,
    'nnt': p.nnt,
    'dnt': p.dnt,

    # model
    'lstm_coef': args.lstm_coef,
    'nlayers': args.lstm_layer,
    'bsz': args.batch_size,
    'dhid': args.lstm_dim,
    'lexicon': p.lexicon,
    'parser': parser,
    'unt_pre': p.unt_pre,
    'p2l_pre': p.p2l_pre,
    'pl2r_pre': p.pl2r_pre,
    'initrange': 1,
}

model = LCNPModel(inputs)
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
    # define the optimizer to use; currently use Adam
    optimizer = optim.Adam(
        parameters, lr=args.learning_rate, weight_decay=args.l2_coef
    )

    total = len(p.lines) / 2
    template = "Epoch {} Batch {} [{}/{} ({:.1f}%)] Loss: {:.4f}" \
        + " Forward: {:.4f} Backward: {:.4f} Optimize: {:.4f}"

    try:
        for epoch in range(args.epochs):
            idx = 0
            batch = 0
            while not idx == -1:
                start = time.time()
                # get next training instances
                idx = p.next(idx, args.batch_size)
                batch += 1
                optimizer.zero_grad()
                # the parameters array
                p_array = [
                    p.sens,
                    p.p2l, p.pl2r_p, p.pl2r_l, p.unt, p.ut,
                    p.p2l_t, p.pl2r_t, p.unt_t, p.ut_t,
                    p.p2l_i, p.pl2r_i, p.unt_i, p.ut_i
                ]

                # create PyTorch Variables
                if args.cuda:
                    p_array = [(Variable(x)).cuda() for x in p_array]
                else:
                    p_array = [Variable(x) for x in p_array]

                # compute loss
                loss = model('supervised', p_array)

                # back propagation
                t0 = time.time()
                loss.backward()
                t1 = time.time()
                # take an optimization step
                optimizer.step()
                end = time.time()
                if args.verbose:
                    print template.format(
                            epoch, batch, idx, total,
                            float(idx)/total * 100., 
                            loss.data[0],
                            round(t0 - start, 5),
                            round(t1 - t0, 5),
                            round(end - t1, 5)
                        )
        torch.save( {'state_dict': model.state_dict()}, file_save )

    except KeyboardInterrupt:
        if args.verbose:
            print " - Exiting from training early"
        torch.save( {'state_dict': model.state_dict()}, file_save )

    if args.verbose:
        print "Finish supervised training"

def unsupervised():
    # get model paramters
    parameters = itertools.ifilter(
        lambda x: x.requires_grad, model.parameters()
    )
    # define the optimizer to use; currently use Adam
    optimizer = optim.Adam(
        parameters, lr=args.learning_rate, weight_decay=args.l2_coef
    )

    total = len(p.lines)
    template = "Epoch {} Batch {} [{}/{} ({:.1f}%)] Loss: {:.4f}" \
        + " Forward: {:.4f} Backward: {:.4f} Optimize: {:.4f}"
    try:
        for epoch in range(args.epochs):
            idx = 0
            batch = 0
            while not idx == -1:
                idx = p.next(idx)
                if not idx == -1:
                    batch += 1
                    start = time.time()
                    optimizer.zero_grad()
                    # create PyTorch Variables
                    if args.cuda:
                        p_sen = Variable(p.sen).cuda()
                    else:
                        p_sen = Variable(p.sen)
                    # compute the loss
                    loss = model(p_sen)

                    if loss.data[0] > 0:
                        # there is a parse
                        t0 = time.time()
                        loss.backward()
                        t1 = time.time()

                        optimizer.step()
                        end = time.time()
                        if args.verbose:
                            print template.format(
                                    epoch, batch, idx, total,
                                    float(idx)/total * 100., 
                                    loss.data[0],
                                    round(t0 - start, 5),
                                    round(t1 - t0, 5),
                                    round(end - t1, 5)
                                )
        torch.save( {'state_dict': model.state_dict()}, file_save )

    except KeyboardInterrupt:
        if args.verbose:
            print " - Exiting from training early"
        torch.save( {'state_dict': model.state_dict()}, file_save )

    if args.verbose:
        print "Finish unsupervised training"

def parse(sentence):
    indices = p.get_idx(sentence)
    if args.cuda:
        sen = Variable(indices).cuda()
    else:
        sen = Variable(indices)

    return model.parse(sentence, sen)

def test():
    # parsing
    start = time.time()
    instances = ptb("train", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=100)
    test = list(instances)
    cumul_accuracy = 0
    num_trees_with_parse = 0
    for (sentence, gold_tree) in test:
        parse_tree = parse(sentence)
        print parse_tree

        tree = oneline(unbinarize(gold_tree))
        if parse_tree != "":
            tree_accruacy = evalb.evalb(tree, unbinarize(Tree.fromstring(parse_tree)))
            cumul_accuracy += tree_accruacy
            num_trees_with_parse += 1
            print tree_accruacy
        else:
            print "No parse!"

    end = time.time()
    print "Parsing takes %.4f secs\n" % round(end - start, 5)
    print "Fraction of trees with parse = %.4f\n" % round(float(num_trees_with_parse) / len(test), 5)
    accuracy = cumul_accuracy / num_trees_with_parse
    print "Parsing accuracy = %.4f\n" % round(accuracy, 5)

## run the model
if args.mode == 'spv_train':
    supervised()
elif args.mode == 'uspv_train':
    unsupervised()
elif args.mode == 'test':
    test()
elif args.mode == 'parse':
    print "Please enter sentences to parse, one per newline (press \"Enter\" to quit):"
    while True:
        sentence = raw_input()
        if sentence == "":
            break
        parse(sentence)
else:
    print "Cannot recognize the mode, allowed modes are: " \
        "spv_train, uspv_train, parse, test"