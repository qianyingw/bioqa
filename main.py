#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:00:34 2020

@author: qwang
"""

import os
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim

# os.chdir('/home/qwang/bioqa')

from arg_parser import get_args
from data_loader import MNDIterators
from model import BiDAF
from train import train, evaluate


#%% Get arguments from command line
args = get_args()

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
MND = MNDIterators(vars(args))
train_data, valid_data, test_data = MND.create_data()
train_iter, valid_iter, test_iter = MND.create_iterators(train_data, valid_data, test_data)
 
#%% Define the model
vocab_size = len(MND.TEXT.vocab)  
unk_idx = MND.TEXT.vocab.stoi[MND.TEXT.unk_token]  # 0
pad_idx = MND.TEXT.vocab.stoi[MND.TEXT.pad_token]  # 1

model = BiDAF(vocab_size = vocab_size,
              embed_dim = args.embed_dim, 
              hidden_dim= args.hidden_dim, 
              num_layers = args.num_layers,
              dropout = args.dropout, 
              pad_idx = pad_idx)

n_pars = sum(p.numel() for p in model.parameters())
# print(model)
print("Number of parameters: {}".format(n_pars))

#%% Load pre-trained embedding
pretrained_embeddings = MND.TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[unk_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <unk> tokens
model.embedding.weight.data[pad_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <pad> tokens

del pretrained_embeddings


#%% Define the optimizer & scheduler
optimizer = optim.Adam(model.parameters(1e-4))   
if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)

# Slanted triangular Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
# total_steps = len(train_loader) * 9 // args.accum_step  # change n_epochs to total #epochs for resuming training
total_steps = len(train_iter) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

for epoch in range(args.num_epochs):   
    train_scores = train(model, train_iter, optimizer, scheduler, args.clip, args.accum_step)
    valid_scores = evaluate(model, MND, valid_iter)        

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
    
    print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))                       
    print('\n[Train] loss: {0:.3f}'.format(train_scores['loss']))
    print('[Val] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}%\n'.format(valid_scores['loss'], valid_scores['em']*100, valid_scores['f1']*100))
    
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)
        

#%% Test
# if args.save_model:
#     test_scores = test(model, test_iterator, criterion, metrics_fn, args, restore_file = 'best')