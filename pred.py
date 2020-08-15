#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 19:57:07 2020

@author: qwang
"""

import json
import pandas as pd
import dill
import re
import json
import os

import spacy
import torch

import utils
from data_loader import MNDIterators
from model import BiDAF

nlp = spacy.load("en_core_web_sm")

#%%
def word_tokenizer(text, lower=True):
    if lower:
        tokens = [token.text.lower() for token in nlp(text)]
    else:
        tokens = [token.text for token in nlp(text)]
    return tokens


def save_field(arg_path, field_path):
    with open(arg_path) as f:
        args = json.load(f)['args']
    args['data_dir'] = '/media/mynewdrive/bioqa/mnd'
    args['embed_path'] = '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt'
    args['exp_dir'] = '/home/qwang/bioqa/exps'
    
    # Build vocab
    MND = MNDIterators(args)
    train_data, valid_data, test_data = MND.create_data()
    MND.build_vocabulary(train_data, valid_data, test_data) 
    TEXT = MND.TEXT
 
    # Save TEXT file
    with open(field_path,"wb") as f:
         dill.dump(TEXT, f)
        
# Save
# save_field(arg_path = '/home/qwang/bioqa/exps/exps_prfs.json',
#            field_path = '/home/qwang/bioqa/exps/exps.Field')

#%%
def pred_answer(context, question, arg_path, field_path, pth_path, device=torch.device('cpu')):
    
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
    
    # Load TEXT field
    with open(field_path,"rb") as fin:
        TEXT = dill.load(fin)   
     
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]  # 1

    # Load model
    model = BiDAF(vocab_size = len(TEXT.vocab),
                  embed_dim = args['embed_dim'], 
                  hidden_dim= args['hidden_dim'], 
                  num_layers = args['num_layers'],
                  dropout = args['dropout'], 
                  pad_idx = pad_idx)
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cpu()
     
    # Load pre-trained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <unk> tokens
    model.embedding.weight.data[pad_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <pad> tokens
    
    # Tokenization
    c_tokens = word_tokenizer(context, lower=False)
    q_tokens = word_tokenizer(question, lower=False)
    
    # Convert tokens into idx
    c_idxs = [TEXT.vocab.stoi[t] for t in c_tokens]
    q_idxs = [TEXT.vocab.stoi[t] for t in q_tokens]    
     
    # Prediction
    model.eval()
    c_tensor = torch.LongTensor(c_idxs).to(device)  # [c_len]
    c_tensor = c_tensor.unsqueeze(0)  # add dim for batch size  
    q_tensor = torch.LongTensor(q_idxs).to(device)   
    q_tensor = q_tensor.unsqueeze(0) 
    
    p1, p2 = model(c_tensor, q_tensor) 
    
    # Get start/end idx
    p1, p2 = p1.exp(), p1.exp()
    s_idx, e_idx = utils.get_ans_idx(p1, p2)   
    s_idx, e_idx = s_idx.item(), e_idx.item()
    
    # Convert answer idxs to tokens
    ans_tokens = []   
    if s_idx == e_idx:       
        ans_vocab_idx = c_idxs[s_idx]  # "idx of content" =>>> "idx of TEXT vocab"
        text = TEXT.vocab.itos[ans_vocab_idx]  # "idx of TEXT vocab" =>>> "answer text token"
        ans_tokens.append(text)
    else:
        ans_vocab_idx = c_idxs[s_idx:e_idx]  
        for vocab_idx in ans_vocab_idx:
            text = TEXT.vocab.itos[vocab_idx]
            ans_tokens.append(text)
            
    ans = " ".join(ans_tokens)
    
    return ans
            


#%% Test (del)
context = """
Delta-9-THC in the treatment of spasticity associated with multiple sclerosis. Marijuana is reported to decrease spasticity in patients with multiple sclerosis. This is a double blind, placebo controlled, crossover clinical trial of delta-9-THC in 13 subjects with clinical multiple sclerosis and spasticity. Subjects received escalating doses of THC in the range of 2.5-15 mg., five days of THC and five days of placebo in randomized order, divided by a two-day washout period. Subjective ratings of spasticity and side effects were completed and semiquantitative neurological examinations were performed. At doses greater than 7.5 mg there was significant improvement in patient ratings of spasticity compared to placebo. These positive findings in a treatment failure population suggest a role for THC in the treatment of spasticity in multiple sclerosis
"""
question ="What is the intervention?"


answer = pred_answer(context, question, 
                     arg_path = '/home/qwang/bioqa/exps/exps_prfs.json',
                     field_path = '/home/qwang/bioqa/exps/exps.Field', 
                     pth_path = '/home/qwang/bioqa/exps/best.pth.tar', 
                     device=torch.device('cpu'))
