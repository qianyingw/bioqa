#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:47:55 2020

@author: qwang
"""


import os

import torch
from torchtext import data
import torchtext.vocab as vocab

        

#%%
class MNDIterators(object):
    
    def __init__(self, args):
        
        self.args = args

        self.ID = data.RawField()
        self.TEXT = data.Field(batch_first=True)
        self.POSITION = data.Field(batch_first=True)        
        

    def create_data(self):
        
        # If a Field is shared between two columns in a dataset (e.g., question/answer in a QA dataset), 
        # then they will have a shared vocabulary.
        fields = {'id': ('id', self.ID), 
                  'ques_tokens': ('question', self.TEXT), 
                  'context_tokens': ('context', self.TEXT),
                  'y1s': ('y1s', self.POSITION),
                  'y2s': ('y2s', self.POSITION)}
        
        assert os.path.exists(self.args['data_dir']), "Path not exist!"     

        train_data, valid_data, test_data = data.TabularDataset.splits(path = self.args['data_dir'],
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
        
        # self.ID.build_vocab(train_data)  # can't build vocab for RawField
        self.POSITION.build_vocab(train_data)
        
        self.TEXT.build_vocab(train_data, valid_data,
                              max_size = self.args['max_vocab_size'],
                              min_freq = self.args['min_occur_freq'],
                              vectors = self.load_embedding(),
                              unk_init = torch.Tensor.normal_)


    def create_iterators(self, train_data, valid_data, test_data):
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')          
        
        self.build_vocabulary(train_data, valid_data, test_data)
        
        train_iterator = data.BucketIterator.splits(
            train_data,
            sort = True,
            sort_key = lambda x: len(x.context),
            shuffle = True,
            batch_size = self.args['batch_size'],
            device = device
        )
        
        valid_iterator, test_iterator = data.BucketIterator.splits(
            (valid_data, test_data),
            sort = False,
            shuffle = False,
            batch_size = self.args['batch_size'],
            device = device
        )
        
        return train_iterator, valid_iterator, test_iterator


#%% Instance
# args = {
#     'batch_size': 32,
#     'max_vocab_size': 30000,
#     'min_occur_freq': 0,
#     'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
#     'data_dir': '/media/mynewdrive/bioqa/mnd'
#     }             

# MND = MNDIterators(args)
# train_data, valid_data, test_data = MND.create_data()
# train_iter, valid_iter, test_iter = MND.create_iterators(train_data, valid_data, test_data)


# MND.load_embedding().stoi['set']  # 347
# MND.load_embedding().stoi['Set']  # 11912
# MND.load_embedding().stoi['SET']  # 32073

# MND.TEXT.vocab.itos[:12]  # ['<unk>', '<pad>', ',', '.', 'the', 'of', 'and', '-', 'in', ')', '(', 'with']
# MND.TEXT.vocab.itos[-4:]  # ['▵', '⩾', '⩾1', '⩾100']

# MND.TEXT.pad_token  # '<pad>'
# MND.TEXT.unk_token  # '<unk>'
# MND.TEXT.vocab.stoi[MND.TEXT.pad_token]  # 1
# MND.TEXT.vocab.stoi[MND.TEXT.unk_token]  # 0
# MND.TEXT.vocab.vectors.shape  # [20878, 200]


# count = 0
# for batch in valid_iter:    
#     if count < 20:
#         print(batch.context.shape)   # [batch_size, context_len]        
#     count += 1

# count = 0
# for batch in valid_iter:    
#     if count < 8:
#         print("=======================")
#         print(batch.context.shape)   # [batch_size, context_len]
#         print(batch.question.shape)  # [batch_size, question_len]
#         print(batch.y1s.shape)    
#         print(batch.context[0,:].shape) 
#         print(batch.context[1,:].shape)  
#         print(batch.context[-1,:].shape)             
#         count += 1
    
# b = next(iter(train_iter))
# vars(b).keys()  # dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'id', 'question', 'context', 'y1s', 'y2s'])
    

