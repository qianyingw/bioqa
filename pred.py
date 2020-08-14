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
import spacy
import os
nlp = spacy.load("en_core_web_sm")


import torch

from model import BiDAF


#%%
def word_tokenizer(text, lower=True):
    if lower:
        tokens = [token.text.lower() for token in nlp(text)]
    else:
        tokens = [token.text for token in nlp(text)]
    return tokens


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
    model = BiDAF(vocab_size = len(TEXT.vocab)  ,
                  embed_dim = args.embed_dim, 
                  hidden_dim= args.hidden_dim, 
                  num_layers = args.num_layers,
                  dropout = args.dropout, 
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
    c_tensor = torch.LongTensor(c_idxs).to(device)
    c_tensor = c_tensor.unsqueeze(0)  # add dim for batch size  
    q_tensor = torch.LongTensor(q_idxs).to(device)   
    q_tensor = q_tensor.unsqueeze(0) 
    
    p1, p2 = model(c_tensor, q_tensor) 
    
    

    
    
    
    
    
    
    