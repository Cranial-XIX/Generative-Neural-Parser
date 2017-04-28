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

def main():

    parser = argparse.ArgumentParser(
        description='Welcome to the generative neural parser'
    )

    # Below are variables associated with files
    # =========================================================================
    parser.add_argument(
        '-train', '--TrainFile', required=False, default=constants.TRAIN,
        help='Train file path'
    )

    parser.add_argument(
        '-pretrain', '--PretrainedFile', required=False,
        help='Pretrained file path'
    )

    parser.add_argument(
        '-corpus', '--UseStoredCorpus', required=False, default="no",
        help='Whether to use stored corpus: True or False'
    )

    parser.add_argument(
        '-dt', '--TerminalDimension', required=False, default=100,
        help='Terminal\'s dimension'
    )

    parser.add_argument(
        '-rd', '--ShouldReadData', required=False, default="yes",
        help='Whether read and process new data'
    )
    
    parser.add_argument(
        '-v', '--Verbose', required=False, default="yes",
        help='Whether to print logging information'
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
        '-lstm_coef', '--CoefLSTM', required=False, default=1.0,
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
    
    parser.add_argument(
        '-cuda', action='store_true',
        help='use CUDA'
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            if args.Verbose == 'yes':
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.Seed)

    # Create folder to save model and log files
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    tag_model = 'PID='+str(id_process)+'_TIME='+time_current

    path_folder = './output/' + tag_model + '/'
    file_save = ""
    if not args.Mode == 'parse':
        os.makedirs(path_folder)
        file_save = os.path.abspath(path_folder + 'model_dict')

    ## show values ##
    if args.Verbose == 'yes':
        print ""
        print ("PID is : %s" % str(id_process) )
        print ("TIME is : %s" % time_current )

        print "- Files:"
        print (" Train file is : %s" % args.TrainFile )
        print (" Pretrained file is : %s" % args.PretrainedFile)
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
        'pretrain': args.PretrainedFile,
        'train': args.TrainFile,
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
        'cuda': args.cuda,
        'verbose': args.Verbose
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

    elif args.Mode == 'parse':
        # parsing
        sen2parse = "DEADBEAF"
        while not sen2parse == "":
            try:
                if args.Verbose == 'yes':
                    sen2parse = raw_input("Please enter the sentence" \
                        "to parse, or press enter to quit the parser: \n")
                else:
                    sen2parse = raw_input()
                if sen2parse == "":
                    break
            except (EOFError):
                break
            controller.parse_LCNP(p, sen2parse, cmd_inp)
    else:
        print "Cannot recognize the mode, should be chosen from " \
            "{spv_train, uspv_train, parse}"

if __name__ == "__main__": main()