import argparse
import time
import torch

import constants
import controller
import data_processor
import gr

import evalb
from ptb import ptb
from nltk import Tree
from util import unbinarize, oneline

def main():

    args_parser = argparse.ArgumentParser(
        description='Welcome to the generative neural parser'
    )

    args_parser.add_argument(
        '-lstm_coef', '--CoefLSTM', required=False, default=1.0,
    )

    args_parser.add_argument(
        '-lstm_layer', '--LayerLSTM', required=False, default=3,
    )

    args_parser.add_argument(
        '-lstm_dim', '--DimLSTM', required=False, default=100,
        help='Dimension of LSTM model '
    )
    
    args_parser.add_argument(
        '-cuda', action='store_true',
        help='use CUDA'
    )

    args = args_parser.parse_args()

    cmd_inp = {
        'coef_lstm': float(args.CoefLSTM),
        'layer_lstm': int(args.LayerLSTM),
        'dim_model': int(args.DimLSTM),
        'cuda': args.cuda,
        
        # filler variables
        'pretrain': None,
        'train': constants.TRAIN,
        'dt': 100,
        'rd': False,
        'verbose': 'yes'
    }

    p = data_processor.Processor(cmd_inp)
    p.read_and_process()
    parser = gr.GrammarObject(p)
    parser.read_gr_file('xbar.grammar')

    # parsing
    begin = time.time()
    examples = ptb("test", minlength=3, maxlength=30)
    test = list(examples)
    cumul_accuracy = 0
    num_trees_with_parse = 0
    for (sentence, gold_tree) in test:
        parse_tree = controller.parse_LCNP(p, parser, sentence, cmd_inp)

        # print "log of Pr( ", "sentence", ") = ", log_prob_sentence
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
    print "Parsing takes %.4f secs\n" % round(end - begin, 5)
    print "Fraction of trees with parse = %.4f\n" % round(float(num_trees_with_parse) / len(test), 5)
    accuracy = cumul_accuracy / num_trees_with_parse
    print "Parsing accuracy = %.4f\n" % round(accuracy, 5)

if __name__=="__main__": main()
