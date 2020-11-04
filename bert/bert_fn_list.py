#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:56:34 2020

Compared with <def train_fn> and <def valid_fn> in bert_fn.py:
	- Use "utils.get_ans_list_idx" instead "utils.get_ans_idx" to get idxs of top [num_answer] candidates
	- Obtain "ans_tokens_pred_list" instead "ans_tokens_pred"
    - Add score['mrr']

@author: qwang
"""
from tqdm import tqdm

import torch
import torch.nn as nn
import transformers

import utils


#%% Train
def train_fn_list(model, data_loader, optimizer, scheduler, tokenizer, clip, accum_step, device):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'mrr': 0}
    len_iter = len(data_loader)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            input_ids = batch['input_ids'].to(device)  # [batch_size, c_len]
            attn_mask = batch['attention_mask'].to(device)  # [batch_size, c_len]
            y1s = batch['token_starts'].to(device)  # [batch_size]
            y2s = batch['token_ends'].to(device)  # [batch_size]
            
            if type(tokenizer) == transformers.tokenization_distilbert.DistilBertTokenizer:
                outputs = model(input_ids, attention_mask = attn_mask, 
                                start_positions = y1s, end_positions = y2s) 
            if type(tokenizer) == transformers.tokenization_distilbert.DistilBertTokenizerFast:
                outputs = model(input_ids, attention_mask = attn_mask, 
                                start_positions = y1s, end_positions = y2s) 
            if type(tokenizer) == transformers.tokenization_bert.BertTokenizerFast:
                outputs = model(input_ids, attention_mask = attn_mask, 
                                start_positions = y1s, end_positions = y2s)   
            if type(tokenizer) == transformers.tokenization_longformer.LongformerTokenizerFast:
                outputs = model(input_ids, attention_mask = attn_mask)   
            
            loss = outputs[0]
            p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]
    
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
            p1s = utils.masked_softmax(p1s, attn_mask, dim=1, log_softmax=True)  # [batch_size, c_len]
            p2s = utils.masked_softmax(p2s, attn_mask, dim=1, log_softmax=True)  # [batch_size, c_len]
            p1s, p2s = p1s.exp(), p1s.exp()
            
            # s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
            s_idxs, e_idxs = utils.get_ans_list_idx(p1s, p2s, num_answer=5)  # [batch_size, num_answer]
            
            for i in range(p1s.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                
                if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                
                    ## Predicted answer tokens
                    # Token list of first answer candidate (for f1/prec/rec)
                    ans_piece_tokens = all_tokens[s_idxs[i,0]: (e_idxs[i,0]+1)]
                    answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                    answer = tokenizer.decode(answer_ids)
                    ans_tokens_pred = answer.lower().split()  
                    # List of token list of [num_answer] answer candidates
                    ans_tokens_pred_list = []
                    for j in range(s_idxs.shape[1]):
                        ans_jth_tokens = all_tokens[s_idxs[i,j]: (e_idxs[i,j]+1)]
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_jth_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_pred_list.append(answer.lower().split())  
                        
                    
                    ## True answer tokens
                    ans_piece_tokens = all_tokens[y1s[i]: (y2s[i]+1)]
                    answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                    answer = tokenizer.decode(answer_ids)
                    ans_tokens_true = answer.lower().split()
                    
                    scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                    f1, prec, rec = utils.metric_f1_pr(ans_tokens_pred, ans_tokens_true) 
                    scores['f1'] += f1
                    scores['prec'] += prec
                    scores['rec'] += rec
                    scores['mrr'] += utils.metric_rr(ans_tokens_pred_list, ans_tokens_true)
                    
                n_samples += 1    
                
            progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples
    scores['mrr'] = scores['mrr'] / n_samples

    return scores



#%% Evaluate
def valid_fn_list(model, data_loader, tokenizer, device):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'mrr': 0}
    len_iter = len(data_loader)
    n_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for j, batch in enumerate(data_loader):                      
                
                input_ids = batch['input_ids'].to(device)  # [batch_size, c_len]
                attn_mask = batch['attention_mask'].to(device)  # [batch_size, c_len]
                y1s = batch['token_starts'].to(device)  # [batch_size]
                y2s = batch['token_ends'].to(device)  # [batch_size]
                
                if type(model) == transformers.modeling_distilbert.DistilBertForQuestionAnswering:
                    outputs = model(input_ids, attention_mask = attn_mask, 
                                    start_positions = y1s, end_positions = y2s) 
                if type(tokenizer) == transformers.tokenization_bert.BertTokenizerFast:
                    outputs = model(input_ids, attention_mask = attn_mask, 
                                    start_positions = y1s, end_positions = y2s)   
                if type(model) == transformers.modeling_longformer.LongformerForQuestionAnswering:
                    outputs = model(input_ids, attention_mask = attn_mask)         
                
                loss = outputs[0]
                p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]  

                # Get start/end idxs
                mask_c = attn_mask != torch.zeros_like(attn_mask)
                p1s = utils.masked_softmax(p1s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p2s = utils.masked_softmax(p2s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p1s, p2s = p1s.exp(), p1s.exp()
                
                # s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
                s_idxs, e_idxs = utils.get_ans_list_idx(p1s, p2s, num_answer=5)  # [batch_size, num_answer]
                

                for i in range(p1s.shape[0]):
                    all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                    
                    if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                    
                        ## Predicted answer tokens
                        # Token list of first answer candidate (for f1/prec/rec)
                        ans_piece_tokens = all_tokens[s_idxs[i,0]: (e_idxs[i,0]+1)]
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_pred = answer.lower().split()  
                        # List of token list of [num_answer] answer candidates
                        ans_tokens_pred_list = []
                        for j in range(s_idxs.shape[1]):
                            ans_jth_tokens = all_tokens[s_idxs[i,j]: (e_idxs[i,j]+1)]
                            answer_ids = tokenizer.convert_tokens_to_ids(ans_jth_tokens)
                            answer = tokenizer.decode(answer_ids)
                            ans_tokens_pred_list.append(answer.lower().split())  
                            
                        # True answer tokens
                        ans_piece_tokens = all_tokens[y1s[i]: (y2s[i]+1)]
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_true = answer.lower().split()
                        
                        scores['em'] += utils.metric_em(ans_tokens_pred, ans_tokens_true)
                        f1, prec, rec = utils.metric_f1_pr(ans_tokens_pred, ans_tokens_true) 
                        scores['f1'] += f1
                        scores['prec'] += prec
                        scores['rec'] += rec
                        scores['mrr'] += utils.metric_rr(ans_tokens_pred_list, ans_tokens_true)

                    n_samples += 1    
                
                scores['loss'] += loss.item() 
                progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples
    scores['mrr'] = scores['mrr'] / n_samples

    return scores