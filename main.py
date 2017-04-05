import argparse
import datetime
import math
import numpy
import os
import pickle
import time
import torch

import constants
import controller
import data_processor

import test

def main():

    parser = argparse.ArgumentParser(
        description='Welcome to the generative neural parser'
    )

    # Below are variables associated with files
    # =========================================================================
    parser.add_argument(
        '-fd', '--FileData', required=False, default=constants.FILE_DATA,
        help='Path of the dataset'
    )

    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model'
    )

    parser.add_argument(
        '-c', '--UseStoredCorpus', required=False,
        help='Whether to use stored corpus: True or False'
    )

    parser.add_argument(
        '-td', '--TerminalDimension', required=False, default=100,
        help='Terminal\'s dimension'
    )

    parser.add_argument(
        '-rd', '--ShouldReadData', required=False, default="yes",
        help='Whether read and process new data'
    )
    # Below are variables associated with model
    # =========================================================================
    parser.add_argument(
        '-m', '--Mode', required=True,
        help='Which mode to run: spv_train, uspv_train, parse'
    )

    parser.add_argument(
        '-s', '--Seed', required=False, default=12345,
        help='Seed of random state'
    )

    parser.add_argument(
        '-lstm_coef', '--CoefLSTM', required=False, default=1,
    )

    parser.add_argument(
        '-lstm_layer', '--LayerLSTM', required=False, default=3,
    )

    parser.add_argument(
        '-lstm_dim', '--DimLSTM', required=False, default=100,
        help='Dimension of LSTM model '
    )

    parser.add_argument(
        '-cl2', '--CoefL2', required=False, default=1e-2,
        help='Coefficient of L2 norm'
    )

    # Below are variables associated with training
    # =========================================================================
    parser.add_argument(
        '-me', '--MaxEpoch', required=False, default=10,
        help='Max epoch number of training'
    )

    parser.add_argument(
        '-bsz', '--BatchSize', required=False, default=2,
        help='Size of mini-batch'
    )

    parser.add_argument(
        '-lr', '--LearningRate', required=False, default=0.02,
        help="Initial learning rate"
    )
    
    parser.add_argument('--cuda', action='store_true',
        help='use CUDA')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.Seed)
        
    # Create folder to save model and log files
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    tag_model = '_PID='+str(id_process)+'_TIME='+time_current

    path_folder = './output/' + tag_model + '/'
    os.makedirs(path_folder)

    file_save = os.path.abspath(path_folder + 'model_dict.tar')

    ## show values ##
    print ""
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )

    print "- Files:"
    print (" FileData is : %s" % args.FileData )
    print (" FilePretrain is : %s" % args.FilePretrain)
    print (" Terminal's dimension is : %s" % args.TerminalDimension)
    print (" Whether read and process new data : %s") % str(args.ShouldReadData)

    print "- Model:"
    print (" You want the %s mode of the model" % args.Mode ) 
    print (" Seed is : %s" % str(args.Seed) )
    print (" LSTM coefficient is : %s" % str(args.CoefLSTM))
    print (" LSTM number of layer is : %s" % str(args.LayerLSTM))
    print (" DimLSTM is : %s" % str(args.DimLSTM) )
    print (" CoefL2 is : %s" % str(args.CoefL2) )

    print "- Train:"
    print (" MaxEpoch is : %s" % str(args.MaxEpoch) )
    print (" BatchSize is : %s" % str(args.BatchSize) )
    print (" InitialLearningRate is : %s" % str(args.LearningRate) )
    print "==================================================================="

    cmd_inp = {
        'pretrain': args.FilePretrain,
        'data': args.FileData,
        'save': file_save,
        'dt': args.TerminalDimension,
        'rd': args.ShouldReadData == "yes",

        'seed_random': int(args.Seed),
        'coef_lstm': float(args.CoefLSTM),
        'layer_lstm': int(args.LayerLSTM),
        'dim_model': int(args.DimLSTM),
        'coef_l2': float(args.CoefL2),

        'max_epoch': int(args.MaxEpoch),
        'batch_size': int(args.BatchSize),
        'learning_rate': float(args.LearningRate),
        'cuda': args.cuda
    }

    p = data_processor.Processor(cmd_inp)
    p.read_and_process()

    if args.Mode == 'spv_train':
        # supervised training
        controller.spv_train_LCNP(p, cmd_inp)
        #test.test_next(p)
    elif args.Mode == 'uspv_train':
        # unsupervsied training
        controller.uspv_train_LCNP(p, cmd_inp)

    else:
        # parsing
        sen2parse = "DEADBEAF"
        while not sen2parse == "":
            sen2parse = raw_input("Please enter the sentence" \
                "to parse, or press enter to quit the parser: \n")
            controller.parse_LCNP(p, sen2parse, cmd_inp)

if __name__ == "__main__": main()