#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:30:49 2020

@author: qwang
"""


from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F


import utils



#%%
def train_fn(model, BaseIter, iterator, optimizer, scheduler, clip, accum_step):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(iterator):                      
            
            # batch.context: [batch_size, c_len]
            # batch.question: [batch_size, q_len]
            # batch.y1s, batch.y2s: list. len = batch_size
            # p1s, p2s: [batch_size, c_len]
            
            # Get start/end idxs
            log_p1s, log_p2s = model(batch.context, batch.question)
            p1s, p2s = log_p1s.exp(), log_p2s.exp()
            s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
                     
            y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
            
            # Sometimes y1s/y2s are outside the model inputs (like -999), need to ignore these terms
            ignored_idx = p1s.shape[1]
            y1s_clamp = torch.clamp(y1s, min=0, max=ignored_idx)  # limit value to [0, max_c_len]. '-999' converted to 0 
            y2s_clamp = torch.clamp(y2s, min=0, max=ignored_idx)

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_idx)
            loss = (loss_fn(p1s, y1s_clamp) + loss_fn(p2s, y2s_clamp)) / 2         
            # loss = F.nll_loss(log_p1s, y1s) + F.nll_loss(log_p2s, y2s)
            scores['loss'] += loss.item() 
             
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (j+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                                       
            for i in range(p1s.shape[0]):                                              
                if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                    context_idxs = batch.context[i]  
                    # Convert answer idxs to tokens
                    ans_tokens_true = utils.ans_idx_to_tokens(context_idxs, y1s[i], y2s[i], BaseIter)
                    ans_tokens_pred = utils.ans_idx_to_tokens(context_idxs, s_idxs[i].item(), e_idxs[i].item(), BaseIter)
                    
                    scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                    f1, prec, rec = utils.metric_f1_pr(ans_tokens_pred, ans_tokens_true) 
                    scores['f1'] += f1
                    scores['prec'] += prec
                    scores['rec'] += rec
                    
                n_samples += 1   
                
            progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores


#%%
def valid_fn(model, BaseIter, iterator):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for batch in iterator:               
                # Get start/end idxs
                log_p1s, log_p2s = model(batch.context, batch.question)
                p1s, p2s = log_p1s.exp(), log_p2s.exp()
                s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
                         
                y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
                
                # Sometimes y1s/y2s are outside the model inputs (like -999), need to ignore these terms
                ignored_idx = p1s.shape[1]
                y1s_clamp = torch.clamp(y1s, min=0, max=ignored_idx)  # limit value to [0, max_c_len]. '-999' converted to 0 
                y2s_clamp = torch.clamp(y2s, min=0, max=ignored_idx)
    
                loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_idx)
                loss = (loss_fn(p1s, y1s_clamp) + loss_fn(p2s, y2s_clamp)) / 2         
                # loss = F.nll_loss(log_p1s, y1s) + F.nll_loss(log_p2s, y2s)
                scores['loss'] += loss.item() 
                
                for i in range(p1s.shape[0]):                                              
                    if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                        context_idxs = batch.context[i]  
                        # Convert answer idxs to tokens
                        ans_tokens_true = utils.ans_idx_to_tokens(context_idxs, y1s[i], y2s[i], BaseIter)
                        ans_tokens_pred = utils.ans_idx_to_tokens(context_idxs, s_idxs[i].item(), e_idxs[i].item(), BaseIter)
                        
                        scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                        f1, prec, rec = utils.metric_f1_pr(ans_tokens_pred, ans_tokens_true) 
                        scores['f1'] += f1
                        scores['prec'] += prec
                        scores['rec'] += rec
        
                    n_samples += 1                                                                                                                     
                progress_bar.update(1)  # update progress bar 
                
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores
