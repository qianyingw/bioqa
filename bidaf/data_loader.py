#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:47:55 2020

@author: qwang
"""


import os
import json

import torch
from torchtext import data
import torchtext.vocab as vocab


def pad_tokens(tokens, max_len):
    if len(tokens) <= max_len:
        tokens = tokens + ['<pad>']*(max_len-len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

#%%
class BaselineIterators(object):
    
    def __init__(self, args):
        
        self.args = args

        self.ID = data.RawField()
        self.TEXT = data.Field(batch_first=True)
        self.POSITION = data.RawField()     
        
    def process_data(self, process_fn, model='bidaf', max_clen=None, max_qlen=None):

        with open(self.args['data_path']) as fin:
            dat = json.load(fin)
        data_dir = os.path.dirname(self.args['data_path'])
        
        # PsyCIPN data
        if os.path.basename(self.args['data_path']).split('-')[0] == 'PsyCIPN':
            dat_train, dat_valid, dat_test = [], [], []
            for ls in dat:
                if ls['group'] == 'train':
                    dat_train.append(ls)
                elif ls['group'] == 'valid':
                    dat_valid.append(ls)
                else:
                    dat_test.append(ls)
         
            train_processed = process_fn(dat_train)
            valid_processed = process_fn(dat_valid)
            test_processed = process_fn(dat_test)  
        
        # MND data
        if os.path.basename(self.args['data_path']).split('-')[0] == 'MND':                        
            train_processed = process_fn(dat['train'])               
            valid_processed = process_fn(dat['valid'])
            test_processed = process_fn(dat['test'])
            
        # Pading over batches (qanet only)
        if model == 'qanet':
            for i, _ in enumerate(train_processed):
                train_processed[i]['context_tokens'] = pad_tokens(train_processed[i]['context_tokens'], max_len=max_clen)
                train_processed[i]['ques_tokens'] = pad_tokens(train_processed[i]['ques_tokens'], max_len=max_qlen)
            for i, _ in enumerate(valid_processed):
                valid_processed[i]['context_tokens'] = pad_tokens(valid_processed[i]['context_tokens'], max_len=max_clen)
                valid_processed[i]['ques_tokens'] = pad_tokens(valid_processed[i]['ques_tokens'], max_len=max_qlen)
            for i, _ in enumerate(test_processed):
                test_processed[i]['context_tokens'] = pad_tokens(test_processed[i]['context_tokens'], max_len=max_clen)
                test_processed[i]['ques_tokens'] = pad_tokens(test_processed[i]['ques_tokens'], max_len=max_qlen)
                
        # Write to train/valid/test json    
        with open(os.path.join(data_dir, 'train.json'), 'w') as fout:
            for ls in train_processed:     
                fout.write(json.dumps(ls) + '\n')
            
        with open(os.path.join(data_dir, 'valid.json'), 'w') as fout:
            for ls in valid_processed:     
                fout.write(json.dumps(ls) + '\n')
        
        with open(os.path.join(data_dir, 'test.json'), 'w') as fout:
            for ls in test_processed:     
                fout.write(json.dumps(ls) + '\n')  
        
        
    def create_data(self):
        
        # If a Field is shared between two columns in a dataset (e.g., question/answer in a QA dataset), 
        # then they will have a shared vocabulary.
        fields = {'id': ('id', self.ID), 
                  'ques_tokens': ('question', self.TEXT), 
                  'context_tokens': ('context', self.TEXT),
                  'y1s': ('y1s', self.POSITION),
                  'y2s': ('y2s', self.POSITION)}
        
        dir_path = os.path.dirname(self.args['data_path'])
        assert os.path.exists(dir_path), "Path not exist!"     

        train_data, valid_data, test_data = data.TabularDataset.splits(path = dir_path,
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
        # self.POSITION.build_vocab(train_data)
        
        self.TEXT.build_vocab(train_data, valid_data,
                              max_size = self.args['max_vocab_size'],
                              min_freq = self.args['min_occur_freq'],
                              vectors = self.load_embedding(),
                              unk_init = torch.Tensor.normal_)


    def create_iterators(self, train_data, valid_data, test_data):
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')          
        
        self.build_vocabulary(train_data, valid_data, test_data)
        
        train_iterator = data.BucketIterator(
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


# %% Instance
# args = {
#     'batch_size': 32,
#     'max_vocab_size': 30000,
#     'min_occur_freq': 0,
#     'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
#     'data_path': "/media/mynewdrive/bioqa/mnd/intervention/MND-Intervention-1983-06Aug20.json"
#     # 'data_path': "/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json"
#     }             

# BaseIter = BaselineIterators(args)
# train_data, valid_data, test_data = BaseIter.create_data()
# train_iter, valid_iter, test_iter = BaseIter.create_iterators(train_data, valid_data, test_data)


# BaseIter.load_embedding().stoi['set']  # 347
# BaseIter.load_embedding().stoi['Set']  # 11912
# BaseIter.load_embedding().stoi['SET']  # 32073

# BaseIter.TEXT.vocab.itos[:12]  # ['<unk>', '<pad>', ',', 'the', 'of', 'in', '.', 'and', ')', '(', 'to', 'a']
# BaseIter.TEXT.vocab.itos[-4:]  # ['~30o', '~Ctrl', '~nd', '~uced']

# BaseIter.TEXT.pad_token  # '<pad>'
# BaseIter.TEXT.unk_token  # '<unk>'
# BaseIter.TEXT.vocab.stoi[BaseIter.TEXT.pad_token]  # 1
# BaseIter.TEXT.vocab.stoi[BaseIter.TEXT.unk_token]  # 0
# BaseIter.TEXT.vocab.vectors.shape  # [26940, 200] / [20851, 200]


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
#         # print(batch.y1s)
#         # print(batch.y2s)
#         print(len(batch.y1s))
#         # print(batch.y1s.shape)    
#         # print(batch.context[0,:].shape) 
#         # print(batch.context[1,:].shape)  
#         # print(batch.context[-1,:].shape)             
#         count += 1
    
# b = next(iter(train_iter))
# vars(b).keys()  # dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'id', 'question', 'context', 'y1s', 'y2s'])
    

