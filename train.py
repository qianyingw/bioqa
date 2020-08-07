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



#%%
def train(model, iterator, criterion, optimizer, scheduler, metrics, clip, accum_step, threshold):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0}
    len_iter = len(iterator)
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for i, batch in enumerate(iterator):
                       
            # batch.context: [batch_size, c_len]
            # batch.question: [batch_size, q_len]
            # batch.y1s, batch.y2s: [batch_size, 3]
            p1s, p2s = model(batch.context, batch.question) 
            # p1s, p2s: [batch_size, c_len]
            
            # Take the 1st match as true answer
            y1, y2 = batch.y1s[:,0], batch.y2s[:,0]
            loss = F.nll_loss(p1s, y1) + F.nll_loss(p2s, y2)
                       
            scores['loss'] += loss.item() 
            # # Average loss of multiple answers
            # loss_candidates = []
            # for i in range(batch.y1s.shape[1]):
            #     y1, y2 = batch.y1s[:,i], batch.y2s[:,i]
            #     if y1 != -999 and y2 != -999:
            #         loss = F.nll_loss(p1s, y1) + F.nll_loss(p2s, y2)
            #         loss_candidates.append(loss)
            # loss = sum(loss_candidates) / len(loss_candidates)      
                
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (i+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)  # update progress bar             
    
    for key, value in scores.items():
        scores[key] = value / len_iter
    
    return scores