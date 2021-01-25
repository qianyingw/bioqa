#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:50:49 2021

@author: qwang
"""


import os
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim

os.chdir('/home/qwang/bioqa')

from arg_parser import get_args
import utils

import helper.helper_psci as helper_psci
import helper.helper_mnd as helper_mnd

from bidaf.data_loader import BaselineIterators
from bidaf.train import train_fn, valid_fn, train_fn_list, valid_fn_list

from qanet.model import QANet

#%% Get arguments
args = get_args()
MAX_CLEN = 512
MAX_QLEN = 10
NUM_BLOCKS_MOD = 2
NUM_ANSWER = 5
ANS_THRES = 0.1

#%% Set random seed and device
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   

# device
if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    print('Use {} GPUs: '.format(torch.cuda.device_count()), device)
elif torch.cuda.device_count() == 1:
    device = torch.device("cuda")
    print('Use 1 GPU: ', device)
else:
    device = torch.device('cpu')     

#%% Load data and create iterators
# args.data_path = "/media/mynewdrive/bioqa/mnd/intervention/MND-Intervention-1983-06Aug20.json" 
# args.data_path = "/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json"
BaseIter = BaselineIterators(vars(args))
if os.path.basename(args.data_path).split('-')[0] == 'PsyCIPN':
    BaseIter.process_data(process_fn = helper_psci.process_for_baseline, model='qanet', max_clen=MAX_CLEN, max_qlen=MAX_QLEN)
if os.path.basename(args.data_path).split('-')[0] == 'MND':
    BaseIter.process_data(process_fn = helper_mnd.process_for_baseline, model='qanet', max_clen=MAX_CLEN, max_qlen=MAX_QLEN)
    
train_data, valid_data, test_data = BaseIter.create_data()
train_iter, valid_iter, test_iter = BaseIter.create_iterators(train_data, valid_data, test_data)

for batch in valid_iter:    
    print(batch.context.shape, batch.question.shape)   # [batch_size, max_clen], [batch_size, max_qlen]        

#%% Define the model
vocab_size = len(BaseIter.TEXT.vocab)  
unk_idx = BaseIter.TEXT.vocab.stoi[BaseIter.TEXT.unk_token]  # 0
pad_idx = BaseIter.TEXT.vocab.stoi[BaseIter.TEXT.pad_token]  # 1

model = QANet(vocab_size = vocab_size,
              embed_dim = args.embed_dim, 
              max_c_len = MAX_CLEN,
              max_q_len = MAX_QLEN,
              hidden_dim= args.hidden_dim, 
              n_block_mod = NUM_BLOCKS_MOD, 
              pad_idx = pad_idx)

# n_pars = sum(p.numel() for p in model.parameters())
# print(model)
# print("Number of parameters: {}".format(n_pars))

#%% Load pre-trained embedding
pretrained_embeddings = BaseIter.TEXT.vocab.vectors

model.embed_inp.embed.weight.data.copy_(pretrained_embeddings)
model.embed_inp.embed.weight.data[unk_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <unk> tokens
model.embed_inp.embed.weight.data[pad_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <pad> tokens

del pretrained_embeddings

#%% Define the optimizer & scheduler
# optimizer = optim.Adam(model.parameters(1e-4))
optimizer = optim.AdamW(model.parameters(), lr=1e-3)  
 
if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)

# Slanted triangular Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_iter) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

# For early stopping
n_worse = 0
min_valid_loss = float('inf')
max_valid_f1 = float('-inf')

for epoch in range(args.num_epochs): 
    if args.type == 'factoid':
        train_scores = train_fn(model, BaseIter, train_iter, optimizer, scheduler, args.clip, args.accum_step)
        valid_scores = valid_fn(model, BaseIter, valid_iter) 
        print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))
        print('[Train] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%'.format(
            train_scores['loss'], train_scores['em']*100, train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
        print('[Valid] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['em']*100, valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
    else:
        train_scores = train_fn_list(model, BaseIter, train_iter, optimizer, scheduler, args.clip, args.accum_step, NUM_ANSWER, ANS_THRES)
        valid_scores = valid_fn_list(model, BaseIter, valid_iter, NUM_ANSWER, ANS_THRES) 
        print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))
        print('[Train] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%'.format(
            train_scores['loss'], train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
        print('[Valid] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))    

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save scores
    # if valid_scores['loss'] < min_valid_loss:
    #     min_valid_loss = valid_scores['loss']    
    is_best = (valid_scores['f1'] > max_valid_f1)
    
    if is_best == True:   
        max_valid_f1 = valid_scores['f1'] 
        utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_scores.json'))
    
    # Save model
    if args.save_model == True:
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                               is_best = is_best, checkdir = args.exp_dir)

    # Early stopping             
    # if valid_scores['loss']-min_valid_loss > 0: # args.stop_c1) and (max_valid_f1-valid_scores['f1'] > args.stop_c2):
    #     n_worse += 1
    # if n_worse == 5: # args.stop_p:
    #     print("Early stopping")
    #     break
        
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)

#%% plot
utils.plot_prfs(prfs_path) 