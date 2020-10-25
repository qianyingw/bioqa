#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:29:15 2020

@author: qwang
"""

import os
import json

import torch
from torch.utils.data import DataLoader


from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

import utils
from arg_parser import get_args
from distil_fn import read_data, char2token_encodings, MNDDataset
from distil_fn import train, evaluate

#%% Read data
args = get_args()

# args = {
#     'num_epochs': 4,
#     'batch_size': 32,
#     'data_dir': '/media/mynewdrive/bioqa/mnd/intervention',
#     'data_name': 'MND-Intervention-1983-06Aug20.json',  
#     'accum_step': 4,
#     'warm_frac': 0.1
#     }    

with open(os.path.join(args.data_dir, args.data_name)) as fin:
    dat = json.load(fin)    

train_contexts, train_questions, train_answers = read_data(dat['train']) 
valid_contexts, valid_questions, valid_answers = read_data(dat['valid']) 
 
#%% Encodings & Dataset & DataLoader  
# Define 'Fast' Tokenizer
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

train_encodings = char2token_encodings(train_contexts, train_questions, train_answers, tokenizer, truncation=False)
valid_encodings = char2token_encodings(valid_contexts, valid_questions, valid_answers, tokenizer, truncation=False)

train_dataset = MNDDataset(train_encodings)
valid_dataset = MNDDataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)


#%% Model & Optimizer & Scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = LongformerForQuestionAnswering.from_pretrained(args.wgts_dir)
# model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-base-4096")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Slanted triangular Learning rate scheduler
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)
print('done')

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

# For early stopping
n_worse = 0
min_valid_loss = float('inf')

for epoch in range(args.num_epochs):   
    train_scores = train(model, train_loader, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
    valid_scores = evaluate(model, valid_loader, tokenizer, device)

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save scores
    if valid_scores['loss'] < min_valid_loss:
        min_valid_loss = valid_scores['loss']    
    # if valid_scores['f1'] > max_valid_f1:
    #     max_valid_f1 = valid_scores['f1'] 
        
    is_best = (valid_scores['loss']-min_valid_loss <= 0) # args.stop_c1) and (max_valid_f1-valid_scores['f1'] <= args.stop_c2)
    if is_best == True:       
        utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_scores.json'))
    
    # Save model
    if args.save_model == True:
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                               is_best = is_best, checkdir = args.exp_dir)
    
    print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))                       
    print('[Train] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}%'.format(train_scores['loss'], train_scores['em']*100, train_scores['f1']*100))
    print('[Valid] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}%\n'.format(valid_scores['loss'], valid_scores['em']*100, valid_scores['f1']*100))
    
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
