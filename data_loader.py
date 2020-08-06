#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:47:55 2020

@author: qwang
"""


import os
import json
import random
import math
from torchtext import data
import torchtext.vocab as vocab
import torch


with open('/media/mynewdrive/bioqa/mnd/MND-Intervention-1983-06Aug20.json') as fin:
    dat = json.load(fin)

    # dict_fields = {'id': ('id', self.RAW),
        #                's_idx': ('s_idx', self.LABEL),
        #                'e_idx': ('e_idx', self.LABEL),
        #                'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
        #                'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}
    
#%%
class MNDIterators(object):
    
    def __init__(self, args):
        
        self.args = args
        # Create data field
        self.ID = data.RawField(is_target = False)
        self.TOKENS = data.Field()
        self.POSITION = data.Field()        
        
        # If a Field is shared between two columns in a dataset (e.g., question and answer in a QA dataset), then they will have a shared vocabulary.
        
    def create_data(self):

        fields = {'id': ('id', self.ID), 
                  'question': ('label', self.TOKENS), 
                  'context': ('text', self.TOKENS),
                  'y1s': ('y1s', self.POSITION),
                  'y2s': ('y2s', self.POSITION)}
            
        train_data, valid_data, test_data = data.TabularDataset.splits(path = os.path.dirname(self.args['data_dir']),
                                                                       train = 'train.json',
                                                                       validation = 'valid.json',
                                                                       test = 'test.json',
                                                                       format = 'json',
                                                                       fields = fields)
        return train_data, valid_data, test_data
        

    def load_embedding(self):
        
        embed_path = self.args['embed_path']        
        custom_embedding = vocab.Vectors(name = os.path.basename(embed_path), 
                                         cache = os.path.dirname(embed_path))

        return custom_embedding
    
    
    def build_vocabulary(self, train_data, valid_data, test_data):
        
        # self.ID.build_vocab(train_data, valid_data, test_data)
        # self.LABEL.build_vocab(train_data)
        self.TOKENS.build_vocab(train_data, valid_data,
                                max_size = self.args['max_vocab_size'],
                                min_freq = self.args['min_occur_freq'],
                                vectors = self.load_embedding(),
                                unk_init = torch.Tensor.normal_)


        # self.CHAR.build_vocab(self.train, self.dev)
        # self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))

    def create_iterators(self, train_data, valid_data, test_data):
        
        self.build_vocabulary(train_data, valid_data, test_data)
        
        ## CUDA
        if torch.cuda.is_available(): 
            device = torch.device('cuda') # torch.cuda.current_device() 
        else:
            device = torch.device('cpu')
        
        
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = self.args['batch_size'],
            device = device
        )
        
        return train_iterator, valid_iterator, test_iterator              