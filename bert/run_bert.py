#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:26:38 2020

@author: qwang
"""

import json
import random
import os
os.chdir('/home/qwang/bioqa')
# import sys
# sys.path[0] = sys.path[0][:-5]

from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering #, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import DataLoader

from arg_parser import get_args
import utils
from bert.bert_fn import read_data, char2token_encodings, EncodingDataset
from bert.bert_fn import train_fn, valid_fn, train_fn_list, valid_fn_list

#%% args
args = get_args()
random.seed(args.seed)
#np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # will be slower  


#%% Load data
# data_path = "/media/mynewdrive/bioqa/mnd/intervention/MND-Intervention-1983-06Aug20.json"
# data_path = "/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json"
# data_path = "/media/mynewdrive/bioqa/PsyCIPN-II-1984-30s-20012021.json",
with open(args.data_path) as fin:
    dat = json.load(fin)   

# PsychoCIPN
if os.path.basename(args.data_path).split('-')[0] == 'PsyCIPN':
    dat_train, dat_valid, dat_test = [], [], []
    for ls in dat:
        if ls['group'] == 'train':
            dat_train.append(ls)
        elif ls['group'] == 'valid':
            dat_valid.append(ls)
        else:
            dat_test.append(ls)
    train_contexts, train_questions, train_answers, train_pids = read_data(dat_train) 
    valid_contexts, valid_questions, valid_answers, valid_pids = read_data(dat_valid) 
    test_contexts, test_questions, test_answers, test_pids = read_data(dat_test) 
else:
    # MND
    dat_train, dat_valid, dat_test = dat['train'], dat['valid'], dat['test']
    train_contexts, train_questions, train_answers = read_data(dat_train) 
    valid_contexts, valid_questions, valid_answers = read_data(dat_valid)
    test_contexts, test_questions, test_answers = read_data(dat_test) 
    
    
#%% Encodings & Dataset & DataLoader  
# Define 'Fast' Tokenizer
if args.pre_wgts == 'distil':
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')   
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')   
elif args.pre_wgts == 'biobert':
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1')  
elif args.pre_wgts == 'pubmed-full':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext') 
elif args.pre_wgts == 'pubmed-abs':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')       
else: # args.pre_wgts == 'bert-base'
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_base')    


train_encodings = char2token_encodings(train_contexts, train_questions, train_answers, tokenizer, truncation=True, max_len=512)
valid_encodings = char2token_encodings(valid_contexts, valid_questions, valid_answers, tokenizer, truncation=True, max_len=512)
test_encodings = char2token_encodings(test_contexts, test_questions, test_answers, tokenizer, truncation=True, max_len=512)

# Add pids for list-type   
if os.path.basename(args.data_path).split('-')[0] == 'PsyCIPN':
    train_encodings.update({'pids': train_pids})
    valid_encodings.update({'pids': valid_pids})
    test_encodings.update({'pids': test_pids})
    
train_dataset = EncodingDataset(train_encodings)
valid_dataset = EncodingDataset(valid_encodings)
test_dataset = EncodingDataset(test_encodings)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

#%% Model & Optimizer & Scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.pre_wgts == 'distil':
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")  
elif args.pre_wgts == 'biobert':
    model = BertForQuestionAnswering.from_pretrained("dmis-lab/biobert-v1.1")
elif args.pre_wgts == 'pubmed-full':
    model = BertForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext') 
elif args.pre_wgts == 'pubmed-abs':
    model = BertForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')        
else: # args.pre_wgts == 'bert-base'
    model = BertForQuestionAnswering.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_base')   
    
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

# Slanted triangular Learning rate scheduler
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

# For early stopping
min_valid_loss = float('inf')
# max_valid_f1 = float('-inf')

for epoch in range(args.num_epochs): 
    if args.type == 'factoid':
        train_scores = train_fn(model, train_loader, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
        valid_scores = valid_fn(model, valid_loader, tokenizer, device)
        print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))
        print('[Train] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%'.format(
            train_scores['loss'], train_scores['em']*100, train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
        print('[Valid] loss: {0:.3f} | em: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['em']*100, valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
    else:
        train_scores = train_fn_list(model, train_loader, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device, args.num_answer, args.ans_thres)
        valid_scores = valid_fn_list(model, valid_loader, tokenizer, device, args.num_answer, args.ans_thres)
        print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))
        print('[Train] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%'.format(
            train_scores['loss'], train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
        print('[Valid] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
        
    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save model
    is_best = (valid_scores['loss'] < min_valid_loss) # (valid_scores['f1'] > max_valid_f1)
    if is_best == True:   
        min_valid_loss = valid_scores['loss']    
        
    if args.save_model == True:
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                               is_best = is_best, checkdir = args.exp_dir)

        
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)

#%% Test
if args.save_model:
    pth_dir = 'bioqa/exps/psci/list/test'
    utils.load_checkpoint(os.path.join(args.exp_dir, 'best.pth.tar'), model)
    test_scores = valid_fn_list(model, test_loader, tokenizer, device, args.num_answer, args.ans_thres)
    
    save_path = os.path.join(args.exp_dir, "test_scores.json")
    utils.save_dict_to_json(test_scores, save_path) 
    print('[Test] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%\n'.format(
    test_scores['loss'], test_scores['f1']*100, test_scores['prec']*100, test_scores['rec']*100)) 

#%% plot
# utils.plot_prfs(prfs_path) 

