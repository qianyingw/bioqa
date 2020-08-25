#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:42:19 2020
Ref: https://huggingface.co/transformers/master/custom_datasets.html#question-answering-with-squad-2-0

@author: qwang
"""


import os
import json
from tqdm import tqdm

from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering  # DistilBertTokenizer
from transformers import AdamW

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



import utils
from arg_parser import get_args



#%%
def read_data(dat):
    '''
        dat: train/valid/test list
    '''
    contexts, questions, answers = [], [], []
    for record in dat:
        context = record['context'].replace("''", '" ').replace("``", '" ') # Replace non-standard quotation marks
        question = record['question'].replace("''", '" ').replace("``", '" ') 
        
        answer = record['answers'][0]  # Use first match as the true answer
        answer['answer_end'] = answer['answer_start'] + len(answer)
        
        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    return contexts, questions, answers
        

def char2token_encodings(contexts, questions, answers, tokenizer):
    '''
        Tokenization
        Convert answer char positions to token positions
    '''
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    token_starts, token_ends = [], []
    for i in range(len(contexts)):
        # Convert character positions to token positions
        token_start = encodings.char_to_token(i, answers[i]['answer_start'])
        token_end = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
        
        token_starts.append(token_start)
        token_ends.append(token_end)
        
        # If None, the answer passage has been truncated
        if token_starts[-1] is None:
            token_starts[-1] = tokenizer.model_max_length
        if token_ends[-1] is None:
            token_ends[-1] = tokenizer.model_max_length
    encodings.update({'token_starts': token_starts, 'token_ends': token_ends})
    
    return encodings
    
#%% Create data encodings 
# Read intervention data
args = get_args()
data_name = 'MND-Intervention-1983-06Aug20.json'
data_dir = '/media/mynewdrive/bioqa/mnd/intervention'

with open(os.path.join(args.data_dir, args.data_name)) as fin:
    dat = json.load(fin)    

# Define 'Fast' Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')    
 
# Data encodings   
train_contexts, train_questions, train_answers = read_data(dat['train']) 
train_encodings = char2token_encodings(train_contexts, train_questions, train_answers, tokenizer)

valid_contexts, valid_questions, valid_answers = read_data(dat['valid']) 
valid_encodings = char2token_encodings(valid_contexts, valid_questions, valid_answers, tokenizer)


#%% Data loader
class MNDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = MNDDataset(train_encodings)
valid_dataset = MNDDataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)


#%% Model & Optimizer & Scheduler
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Slanted triangular Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
# total_steps = len(train_loader) * 9 // args.accum_step  # change n_epochs to total #epochs for resuming training
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Train
def train(model, data_loader, optimizer, scheduler, clip, accum_step):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0}
    len_iter = len(data_loader)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for i, batch in enumerate(data_loader):                      
            
            input_ids = batch['input_ids'].to(device)  # [batch_size, c_len]
            attn_mask = batch['attention_mask'].to(device)  # [batch_size, c_len]
            y1s = batch['token_starts'].to(device)  # [batch_size]
            y2s = batch['token_ends'].to(device)  # [batch_size]
            
            outputs = model(input_ids, 
                            attention_mask = attn_mask, 
                            start_positions = y1s, 
                            end_positions = y2s)        
            
            loss = outputs[0]
            p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]
    
            scores['loss'] += loss.item() 
             
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (i+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
                
            # Get start/end idxs
            mask_c = attn_mask != torch.zeros_like(attn_mask)
            p1s = utils.masked_softmax(p1s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
            p2s = utils.masked_softmax(p2s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
            p1s, p2s = p1s.exp(), p1s.exp()
            s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
            
            ans_tokens_pred, ans_tokens_true = [], []
            for i in range(p1s.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                
                # Predicted answer tokens
                ans_piece_tokens = all_tokens[s_idxs[i]: (e_idxs[i]+1)]
                answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                answer = tokenizer.decode(answer_ids)
                ans_tokens = answer.lower().split()
                ans_tokens_pred.append(ans_tokens)
                
                # True answer tokens
                ans_piece_tokens = all_tokens[y1s[i]: (y2s[i]+1)]
                answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                answer = tokenizer.decode(answer_ids)
                ans_tokens = answer.lower().split()
                ans_tokens_true.append(ans_tokens)
                
                scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                scores['f1'] += utils.metric_f1(ans_tokens_pred, ans_tokens_true)   
                n_samples += 1    
                
            progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples

    return scores


#%% Train
model.train()



import time


epochs = 3



for epoch in range(epochs):
    i = 0
    start_time = time.time()
    for batch in train_loader:
        
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        token_starts = batch['token_starts'].to(device)
        token_ends = batch['token_ends'].to(device)
        
        outputs = model(input_ids, attention_mask=attn_mask, start_positions=token_starts, end_positions=token_ends)
        
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        start_logits = outputs[1]
        end_logits = outputs[2]
        i = i+1
        if i % 10 == 0:
            print("Batch: ", i)
       
        
    elapsed_time = time.time() - start_time
    print("Loss: ", loss/len(train_loader))
    print("Time elapsed: ", elapsed_time)



# model.eval()








#%%
args = {
    'batch_size': 32,
    'max_vocab_size': 30000,
    'min_occur_freq': 0,
    'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
    'data_dir': '/media/mynewdrive/bioqa/mnd/intervention'
    }    




