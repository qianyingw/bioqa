#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:57:47 2020

@author: qwang
"""


import argparse
import json
import os

USER = os.getenv('USER')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    parser = argparse.ArgumentParser(description='RoB training and inference helper script')
 
    # Experiments
    parser.add_argument('--seed', nargs="?", type=int, default=1234, help='Seed for random number generator')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=20, help='Number of epochs')    
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file') 
    parser.add_argument('--exp_dir', nargs="?", type=str, default="/home/qwang/bioqa/exps/pci/distil", help='Folder of the experiment')
    parser.add_argument('--clip', nargs="?", type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--accum_step', nargs="?", type=int, default=4, help='Number of steps for gradient accumulation')
    parser.add_argument('--warm_frac', nargs="?", type=float, default=0.1, help='Fraction of iterations when lr increased')
    parser.add_argument('--save_model', nargs="?", type=str2bool, default=False, help='Save model.pth.tar with best loss')   
       
    # Data and embedding
    parser.add_argument('--data_path', nargs="?", type=str,
                        default="/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json",
                        # default="/media/mynewdrive/bioqa/mnd/intervention/MND-Intervention-1983-06Aug20.json", 
                        help='Path of json data')
    parser.add_argument('--pre_wgts', nargs="?", type=str, default="distil", 
                        choices=['distil', 'bert', 'biobert', 'pubmed-full', 'pubmed-abs'],
                        help='Pre-trained model name')
    parser.add_argument('--embed_path', nargs="?", type=str, default="/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt", help='Path of pre-trained vectors')    
    parser.add_argument('--embed_dim', nargs="?", type=int, default=200, help='Dimension of pre-trained word vectors')
    parser.add_argument('--max_vocab_size', nargs="?", type=int, default=30000, help='Maximum size of the vocabulary')
    parser.add_argument('--min_occur_freq', nargs="?", type=int, default=0, help='Minimum frequency of including a token in the vocabulary')
    
      
    # Model
    parser.add_argument('--hidden_dim', nargs="?", type=int, default=64, help='Number of features in RNN hidden state')
    parser.add_argument('--num_layers', nargs="?", type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--dropout', nargs="?", type=float, default=0.5, help='Dropout rate')

   
    args = parser.parse_args()    
    
    if args.args_json_path is None:
        arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
        print(arg_str)
    else:
        args = extract_args_from_json(json_file_path=args.args_json_path, existing_args_dict=args)   
    
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as fin:
        args_dict = json.load(fp=fin)

    for key, value in vars(existing_args_dict).items():
        if key not in args_dict:
            args_dict[key] = value

    args_dict = AttributeAccessibleDict(args_dict)

    return args_dict