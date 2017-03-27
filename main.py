import argparse
import datetime
import math
import numpy
import os
import pickle
import time

import constants
import data_processor
import controller

def main():

    parser = argparse.ArgumentParser(
        description='Welcome to the generative neural parser'
    )

    # Below are variables associated with files
    # =========================================================================
    parser.add_argument(
        '-fd', '--FileData', required=False,
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
        '-lstm', '--CoefLSTM', required=False, default=1,
    )

    parser.add_argument(
        '-d', '--DimLSTM', required=False, default=100,
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

    args = parser.parse_args()


    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    # Create folder to save model and log files
    tag_model = '_PID='+str(id_process)+'_TIME='+time_current


    path_track = './output/' + tag_model + '/'
    log_file = os.path.abspath(
        path_track + 'log.txt'
    )

    path_save = log_file
    command_mkdir = 'mkdir -p ' + os.path.abspath(
        path_track
    )
    os.system(command_mkdir)

    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )

    print "Files:"
    print ("FileData is : %s" % args.FileData )
    print ("FilePretrain is : %s" % args.FilePretrain)

    print "\n"

    print "Model:"
    print ("You want the %s mode of the model" % args.Mode ) 
    print ("Seed is : %s" % str(args.Seed) )
    print ("LSTM coefficient is : %s" % str(args.CoefLSTM))
    print ("DimLSTM is : %s" % str(args.DimLSTM) )
    print ("CoefL2 is : %s" % str(args.CoefL2) )

    print "\n"

    print "Train:"
    print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("BatchSize is : %s" % str(args.BatchSize) )
    print ("InitialLearningRate is : %s" % str(args.LearningRate) )
    print "==================================================================="
    
    dict_args = {
        'PID': id_process,
        'TIME': time_current,

        'FileData': args.FileData,
        'FilePretrain': args.FilePretrain,

        'Seed': args.Seed,
        'CoefLSTM': args.CoefLSTM,
        'DimLSTM': args.DimLSTM,
        'CoefL2': args.CoefL2,

        'MaxEpoch': args.MaxEpoch,
        'BatchSize': args.BatchSize,
        'LearningRate': args.LearningRate
    }

    input_model = {
        'path_pre_train': args.FilePretrain,
        'path_rawdata': args.FileData,      
        'save_file_path': path_save,
        'log_file': log_file,

        'seed_random': args.Seed,
        'coef_lstm': args.CoefLSTM,
        'dim_model': args.DimLSTM,
        'coef_l2': args.CoefL2,

        'max_epoch': args.MaxEpoch,
        'batch_size': args.BatchSize,
        'learning_rate': args.LearningRate,

        'args': dict_args
    }

    processor = data_processor.Processor()
    processor.read_and_process()

    if args.Mode == 'spv_train':
        controller.spv_train_LCNP(processor, input_model)
    elif args.Mode == 'uspv_train':
        controller.uspv_train_LCNP(processor, input_model)
    else:
        #data_processor.read_and_process()
        sen_to_parse = "we 're about to see if advertising works ."
        while sen_to_parse == "":
            sen_to_parse = raw_input("Please enter the sentence to parse: \n")

        controller.parse_LCNP(processor, constants.PRE_TRAINED_FILE, sen_to_parse, input_model)

if __name__ == "__main__": main()