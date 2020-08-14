#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:30:49 2020

@author: qwang
"""


import os
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F


import utils



#%%
def train(model, MND, iterator, optimizer, scheduler, clip, accum_step):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for i, batch in enumerate(iterator):                      
            
            p1s, p2s = model(batch.context, batch.question) 
            # batch.context: [batch_size, c_len]
            # batch.question: [batch_size, q_len]
            # batch.y1s, batch.y2s: [batch_size, 3]
            # p1s, p2s: [batch_size, c_len]
            
            # Take the 1st match as true answer
            y1s, y2s = [], []
            for y1, y2 in zip(batch.y1s, batch.y2s):
                y1s.append([y1[0]])
                y2s.append([y2[0]])
            y1s = torch.tensor(y1s).squeeze(1) 
            y2s = torch.tensor(y2s).squeeze(1)  
            
            # y1, y2 = batch.y1s[:,0], batch.y2s[:,0]
            loss = F.nll_loss(p1s, y1s) + F.nll_loss(p2s, y2s)
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
            p1s, p2s = p1s.exp(), p1s.exp()
            s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
            
            for i, (con, s_pred, e_pred, s_true, e_true) in enumerate(zip(batch.context, s_idxs, e_idxs, y1s, y2s)):  
                
                s_pred, e_pred = s_pred.item(), e_pred.item()
                s_true, e_true = s_true.item(), e_true.item()                    
                # Convert answer idxs to tokens
                ans_tokens_true = utils.ans_idx_to_tokens(con, s_true, e_true, MND)
                ans_tokens_pred = utils.ans_idx_to_tokens(con, s_pred, e_pred, MND)
                # Calculate em/f1 for each record
                scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                scores['f1'] += utils.metric_f1(ans_tokens_pred, ans_tokens_true)   
                n_samples += 1      
                
            progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples

    return scores


#%%
def evaluate(model, MND, iterator):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for batch in iterator:
                           
                p1s, p2s = model(batch.context, batch.question)  # [batch_size, c_len]
                
                # Take the 1st match as true answer
                y1s, y2s = [], []
                for y1, y2 in zip(batch.y1s, batch.y2s):
                    y1s.append([y1[0]])
                    y2s.append([y2[0]])
                y1s = torch.tensor(y1s).squeeze(1)  
                y2s = torch.tensor(y2s).squeeze(1)   
            
                # y1, y2 = batch.y1s[:,0], batch.y2s[:,0]  # [batch_size]
                loss = F.nll_loss(p1s, y1s) + F.nll_loss(p2s, y2s)
                 
                # Get start/end idxs
                p1s, p2s = p1s.exp(), p1s.exp()
                s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
    
                for i, (con, s_pred, e_pred, s_true, e_true) in enumerate(zip(batch.context, s_idxs, e_idxs, y1s, y2s)):
                    
                    s_pred, e_pred = s_pred.item(), e_pred.item()
                    s_true, e_true = s_true.item(), e_true.item()
                    
                    # Convert answer idxs to tokens
                    ans_tokens_true = utils.ans_idx_to_tokens(con, s_true, e_true, MND)
                    ans_tokens_pred = utils.ans_idx_to_tokens(con, s_pred, e_pred, MND)
                    # Calculate em/f1 for each record
                    scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                    scores['f1'] += utils.metric_f1(ans_tokens_pred, ans_tokens_true)   
                    n_samples += 1
                                                                                               
                scores['loss'] += loss.item()                          
                progress_bar.update(1)  # update progress bar 
                
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples

    return scores


