#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:30:49 2020

@author: qwang
"""


from tqdm import tqdm
import itertools

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
            
            y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
            loss, p1s, p2s = model(batch.context, batch.question, y1s, y2s)
            scores['loss'] += loss.item() 
            
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (j+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Get start/end idxs
            s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
                                       
            for i in range(y1s.shape[0]):                                              
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
                
                y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
                loss, p1s, p2s = model(batch.context, batch.question, y1s, y2s)
                scores['loss'] += loss.item() 
            
                # Get start/end idxs
                s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]                
                
                for i in range(y1s.shape[0]):                                              
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
def train_fn_list(model, BaseIter, iterator, optimizer, scheduler, clip, accum_step, num_answer, ans_thres):
    
    scores = {'loss': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    epoch_dic = []
    with tqdm(total=len_iter) as progress_bar:      
        for k, batch in enumerate(iterator):                      
            
            # batch.context, p1s, p2s: [batch_size, c_len]
            # batch.question: [batch_size, q_len]
            # batch.y1s, batch.y2s, batch.pids: [batch_size]     
            y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
            loss, p1s, p2s = model(batch.context, batch.question, y1s, y2s)
            scores['loss'] += loss.item() 
            
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (k+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                       
            # Get start/end idxs. Each record has [num_answer] candidates         
            s_idxs, e_idxs, top_probs = utils.get_ans_list_idx(p1s, p2s, num_answer=num_answer)  # [batch_size, num_answer]

            for i in range(y1s.shape[0]):                                              
                if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                
                    # Context
                    context_idxs = batch.context[i]  
                    # Convert true idxs to tokens for each record
                    ans_tokens_true = utils.ans_idx_to_tokens(context_idxs, y1s[i], y2s[i], BaseIter)
                                        
                    # Convert pred idxs to tokens for each record
                    record_preds_cand = []
                    for j in range(num_answer):  # iter candidates                      
                        ans_tokens_pred = utils.ans_idx_to_tokens(context_idxs, s_idxs[i,j].item(), e_idxs[i,j].item(), BaseIter)
                        record_preds_cand.append(ans_tokens_pred)  # list, len(record_preds_cand)=num_answer
                    
                    record = {'pid': batch.pid[i],  # 1234
                              'trues': ans_tokens_true,    # ["ab", "ef"]
                              'preds': record_preds_cand,  # [["af", "c"], ["b"], ..., ["fg", "ab"]]
                              'probs': top_probs[i].tolist()}  # [0.3, 0.6, ..., 0.1]
                    
                    epoch_dic.append(record)
                n_samples += 1                   
            progress_bar.update(1)  # update progress bar             
                
    # Group epoch_dic by pid
    key = lambda d: d['pid']
    epoch_dic.sort(key=key)
    
    epoch_dic_gp = []
    for key, group in itertools.groupby(epoch_dic, key=key):
        trues, preds, probs = [], [], [] 
        for g in group:
            trues.append(g['trues'])  # [["ab", "ef"], ["cd"]]
            preds = preds + g['preds']  # g['preds']: [["ab","ef"], ["cd"], ["fg","b"], ["b","c"], ["gh"]]
            probs = probs + g['probs']  # g['probs']: [0.3, 0.4, 0.2, 0.7, 0.1]
        # Sort candidates by probs 
        probs, preds = (list(t) for t in zip(*sorted(zip(probs, preds), key=lambda x: x[0], reverse=True)))
        # Keep candidates with probs > thres
        preds = [preds[i] for i in range(len(probs)) if probs[i] > ans_thres]
        # Remove duplicate candidates        
        preds_ndup = []
        for p in preds:
            if p not in preds_ndup:
                preds_ndup.append(p)
     
        epoch_dic_gp.append({'pid': key, 'trues': trues, 'preds': preds_ndup})
            
    for ep in epoch_dic_gp:
        if len(ep['preds']) > 0:
            f1, pre, rec = utils.metric_ave_fpr(ep['preds'], ep['trues'])
            scores['f1'] += f1
            scores['prec'] += pre
            scores['rec'] += rec
        
    scores['loss'] = scores['loss'] / len_iter
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores

#%%
def valid_fn_list(model, BaseIter, iterator, num_answer, ans_thres):
    
    scores = {'loss': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(iterator)
    n_samples = 0
    
    model.eval()
    
    epoch_dic = []
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for batch in iterator:   
                
                y1s, y2s = torch.LongTensor(batch.y1s), torch.LongTensor(batch.y2s)
                loss, p1s, p2s = model(batch.context, batch.question, y1s, y2s)
                scores['loss'] += loss.item() 
            
                # Get start/end idxs
                s_idxs, e_idxs, top_probs = utils.get_ans_list_idx(p1s, p2s, num_answer=num_answer)  # [batch_size, num_answer]                
                
                for i in range(y1s.shape[0]):                                              
                    if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated                    
                        # Context
                        context_idxs = batch.context[i]  
                        # Convert true idxs to tokens for each record
                        ans_tokens_true = utils.ans_idx_to_tokens(context_idxs, y1s[i], y2s[i], BaseIter)
                                            
                        # Convert pred idxs to tokens for each record
                        record_preds_cand = []
                        for j in range(num_answer):  # iter candidates                      
                            ans_tokens_pred = utils.ans_idx_to_tokens(context_idxs, s_idxs[i,j].item(), e_idxs[i,j].item(), BaseIter)
                            record_preds_cand.append(ans_tokens_pred)  # list, len(record_preds_cand)=num_answer
                        
                        record = {'pid': batch.pid[i],  # 1234
                                  'trues': ans_tokens_true,    # ["ab", "ef"]
                                  'preds': record_preds_cand,  # [["af", "c"], ["b"], ..., ["fg", "ab"]]
                                  'probs': top_probs[i].tolist()}  # [0.3, 0.6, ..., 0.1]                       
                        epoch_dic.append(record)       
                    n_samples += 1                                                                                                                     
                progress_bar.update(1)  # update progress bar 
                
    # Group epoch_dic by pid
    key = lambda d: d['pid']
    epoch_dic.sort(key=key)
    
    epoch_dic_gp = []
    for key, group in itertools.groupby(epoch_dic, key=key):
        trues, preds, probs = [], [], []
        for g in group:
            trues.append(g['trues'])  # [["ab", "ef"], ["cd"]]
            preds = preds + g['preds']  # g['preds']: [["ab","ef"], ["cd"], ["fg","b"], ["b","c"], ["gh"]]
            probs = probs + g['probs']  # g['probs']: [0.3, 0.4, 0.2, 0.7, 0.1]
        # Sort candidates by probs 
        probs, preds = (list(t) for t in zip(*sorted(zip(probs, preds), key=lambda x: x[0], reverse=True)))
        # Keep candidates with probs > thres
        preds = [preds[i] for i in range(len(probs)) if probs[i] > ans_thres]
        # Remove duplicate candidates
        preds_ndup = []
        for p in preds:
            if p not in preds_ndup:
                preds_ndup.append(p)
       
        epoch_dic_gp.append({'pid': key, 'trues': trues, 'preds': preds})
            
    for ep in epoch_dic_gp:
        if len(ep['preds']) > 0:
            f1, pre, rec = utils.metric_ave_fpr(ep['preds'], ep['trues'])
            scores['f1'] += f1
            scores['prec'] += pre
            scores['rec'] += rec       
            
    scores['loss'] = scores['loss'] / len_iter
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores