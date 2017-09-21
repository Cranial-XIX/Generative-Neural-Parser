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
    '--train', default=constants.TRAIN_FILE, help='Train file path'
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
    '--model', default="BLN", help='Model: BLN, LN, BSN, BS'
)

argparser.add_argument(
    '--mode', default="parse", help='Mode: spv_train, uspv_train, test, parse'
)

argparser.add_argument(
    '--seed', default=419, help='Random seed'
)

argparser.add_argument(
    '--lstm-coef', default=1.0, help='LSTM coefficient'
)

argparser.add_argument(
    '--lstm-layer', default=3, help='# LSTM layer'
)

argparser.add_argument(
    '--lstm-dim', default=150, help='LSTM hidden dimension'
)

argparser.add_argument(
    '--l2-coef', default=0.02, help='l2 norm coefficient'
)

argparser.add_argument(
    '--verbose', default="yes", help='use verbose mode'
)

# Below are variables associated with training
# =========================================================================
argparser.add_argument(
    '--epochs', default=20, help='# epochs to train'
)

argparser.add_argument(
    '--batch-size', default=15, help='# instances in a batch'
)

argparser.add_argument(
    '--learning-rate', default=0.001, help="learning rate"
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
    print template.format(" train file :", args.train)
    print template.format(" pretrained file :", args.pretrain)
    print template.format(" whether to read new data :", args.read_data)
    print "- Model:"
    print template.format(" model :", args.model)
    print template.format(" model mode :", args.mode)
    print template.format(" LSTM coefficient :", args.lstm_coef)
    print template.format(" LSTM # of layer :", args.lstm_layer)
    print template.format(" LSTM dimension :", args.lstm_dim)
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
args.l2_coef = float(args.l2_coef)
args.make_train = (args.make_train == 'yes')
args.read_data = (args.read_data == 'yes')
args.seed = int(args.seed)
args.verbose = (args.verbose == 'yes')


# let the processor read in data
# Model: BLN, LN, BSN, BS
if args.model == 'BLN':
    dp = PBLN(args.train, args.make_train, args.read_data, args.verbose)

elif args.model == 'LN':
    dp = PLN(args.train, args.make_train, args.read_data, args.verbose)

elif args.model == 'BSN' or args.model == 'BS':
    dp = Processor(args.train, args.make_train, args.read_data, args.verbose)

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

    # define the optimizer to use; currently use Adam
    optimizer = optim.Adam(
        parameters, lr=args.learning_rate, weight_decay=args.l2_coef
    )

    total = dp.trainset_length

    template = "Epch {} Bch {} [{}/{} ({:.1f}%)] NLL: {:.2f}" \
        + " Fwd: {:.2f} Bckwd: {:.2f} Opt: {:.2f}"

    max_accuracy = 0

    for epoch in range(args.epochs):
        dp.shuffle()
        idx = 0
        batch = 0
        tot_loss = 0
        while True:
            batch += 1
            optimizer.zero_grad()

            start = time.time()
            # get next training instances
            idx, next_bch = dp.next(idx, args.batch_size)

            # create PyTorch Variables
            if args.cuda:
                next_bch = [(Variable(torch.LongTensor(x))).cuda() for x in next_bch]
            else:
                next_bch = [Variable(torch.LongTensor(x)) for x in next_bch]

            # compute loss
            loss = model('supervised', next_bch)
            tot_loss += loss
            # back propagation
            t0 = time.time()
            loss.backward()
            t1 = time.time()
            # take an optimization step
            optimizer.step()
            end = time.time()
            if args.verbose:
                if idx == -1:
                    print template.format(
                            epoch, batch, total, total,
                            100.,
                            loss.data[0],
                            round(t0 - start, 5),
                            round(t1 - t0, 5),
                            round(end - t1, 5)
                        )
                    break
                else:
                    print template.format(
                            epoch, batch, idx, total,
                            float(idx)/total * 100.,
                            loss.data[0],
                            round(t0 - start, 5),
                            round(t1 - t0, 5),
                            round(end - t1, 5)
                        )

        print " -- E[ NLL(sentence) ]={}".format(tot_loss.data[0] / total)

        if epoch < 5:
            continue

        avg_accuracy = test()
        model.init_h0()
        if avg_accuracy > max_accuracy:
            torch.save( {'state_dict': model.state_dict()}, file_save )

    if args.verbose:
        print "Finish supervised training"


def parse(sentence):
    sen = Variable( dp.get_idx(sentence) )

    if args.cuda:
        sen = sen.cuda()

    return model.parse(sentence, sen)


def test():
    model.parse_setup()
    test_data = list(ptb("dev", minlength=3, maxlength=constants.MAX_SEN_LENGTH, n=100))
    cumul_accuracy = 0
    num_sen = 0

    for (sentence, gold) in test_data:
        if len(sentence.split()) > 15:
            continue

        num_sen += 1

        parse_string = parse(sentence)

        parse_tree = Tree.fromstring(parse_string)
        tree_accruacy = evalb.evalb(
            oneline(unbinarize(gold)),
            unbinarize(parse_tree)
        )
        cumul_accuracy += tree_accruacy
        print " Accuracy : ", tree_accruacy
        print " GOLD \n",unbinarize(gold).pretty_print()
        print " PARSE \n",unbinarize(parse_tree).pretty_print()
        print " PARSE \n",parse_tree.pretty_print()

    avg_accuracy = cumul_accuracy / float(num_sen)
    print " # Sen: {} Accuracy: {}".format( num_sen, avg_accuracy )
    return avg_accuracy


def check_spv():
    pass

## run the model
if args.mode == 'spv_train':
    supervised()

elif args.mode == 'uspv_train':
    unsupervised()

elif args.mode == 'test':
    test()

elif args.mode == 'parse':
    model.parsing_setup()
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