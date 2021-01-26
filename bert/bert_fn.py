#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:42:19 2020
Ref: https://huggingface.co/transformers/master/custom_datasets.html#question-answering-with-squad-2-0

@author: qwang
"""

from tqdm import tqdm
import itertools

import torch
import torch.nn as nn

import utils

import transformers

#%%
def read_data(dat):
    '''
        dat: train/valid/test list
    '''
    
    is_mnd = False
    if'PubID' not in dat[0].keys():
        is_mnd = True
   
    contexts, questions, answers, pids = [], [], [], []
    for record in dat:
        context = record['context'].replace("''", '" ').replace("``", '" ') # Replace non-standard quotation marks
        question = record['question'].replace("''", '" ').replace("``", '" ') 
        
        if is_mnd == True:
            answer = record['answers'][0]
        else:
            answer = record['answers']
            pids.append(record['PubID'])
        # answer['answer_end'] = answer['answer_start'] + len(answer) - 1       
        answer['answer_end'] = answer['answer_start'] + len(answer['text']) - 1  
        
        contexts.append(context)
        questions.append(question)
        answers.append(answer)
    if is_mnd == True:
        return contexts, questions, answers
    else:
        return contexts, questions, answers, pids
        

#%%
def char2token_encodings(contexts, questions, answers, tokenizer, truncation, max_len):
    '''
        Tokenization
        Convert answer char positions to token positions
    ''' 
    encodings = tokenizer(contexts, questions, truncation=truncation, max_length=max_len, padding=True)
    
    token_starts, token_ends = [], []
    for i in range(len(contexts)):
        # Convert character positions to token positions
        if answers[i]['answer_start'] == -999:  
        # -999 --> answer not found in the 'new context' generated from most [max_n_sent] similar sents (for psycipn data)
            token_starts.append(-999)
            token_ends.append(-999)
        else:     
            token_start = encodings.char_to_token(i, answers[i]['answer_start'])
            token_end = encodings.char_to_token(i, answers[i]['answer_end'])
            
            token_starts.append(token_start)
            token_ends.append(token_end)
        
        # If None, the answer passage has been truncated
        if token_starts[-1] is None:
            token_starts[-1] = -999 # tokenizer.model_max_length
        if token_ends[-1] is None:
            token_ends[-1] = -999 # tokenizer.model_max_length
    encodings.update({'token_starts': token_starts, 'token_ends': token_ends})
    
    return encodings
    


#%% Data loader
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


#%% Train
def train_fn(model, data_loader, optimizer, scheduler, tokenizer, clip, accum_step, device):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0}
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
            
            outputs = model(input_ids, attention_mask = attn_mask, 
                            start_positions = y1s, end_positions = y2s)   
            
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
            s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
            
            # ans_tokens_pred, ans_tokens_true = [], []
            for i in range(p1s.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                
                if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                    # Predicted answer tokens
                    ans_piece_tokens = all_tokens[s_idxs[i]: (e_idxs[i]+1)]
                    answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                    answer = tokenizer.decode(answer_ids)
                    ans_tokens_pred = answer.lower().split()             
                    
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
                    
                n_samples += 1    
                
            progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores



#%% Evaluate
def valid_fn(model, data_loader, tokenizer, device):
    
    scores = {'loss': 0, 'em': 0, 'f1': 0, 'prec': 0, 'rec': 0}
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
                
                outputs = model(input_ids, attention_mask = attn_mask, 
                                start_positions = y1s, end_positions = y2s)        
                
                loss = outputs[0]
                p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]  

                # Get start/end idxs
                mask_c = attn_mask != torch.zeros_like(attn_mask)
                p1s = utils.masked_softmax(p1s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p2s = utils.masked_softmax(p2s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p1s, p2s = p1s.exp(), p1s.exp()
                s_idxs, e_idxs = utils.get_ans_idx(p1s, p2s)  # [batch_size]
                
                ans_tokens_pred, ans_tokens_true = [], []
                for i in range(p1s.shape[0]):
                    all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                    
                    if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                        # Predicted answer tokens
                        ans_piece_tokens = all_tokens[s_idxs[i]: (e_idxs[i]+1)]
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_pred = answer.lower().split()             
                        
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

                    n_samples += 1    
                
                scores['loss'] += loss.item() 
                progress_bar.update(1)  # update progress bar             
            
    scores['loss'] = scores['loss'] / len_iter
    scores['em'] = scores['em'] / n_samples
    scores['f1'] = scores['f1'] / n_samples
    scores['prec'] = scores['prec'] / n_samples
    scores['rec'] = scores['rec'] / n_samples

    return scores

#%% 
def train_fn_list(model, data_loader, optimizer, scheduler, tokenizer, clip, accum_step, device, num_answer, ans_thres):
    
    scores = {'loss': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(data_loader)
    n_samples = 0
    
    model.train()
    optimizer.zero_grad()
    
    epoch_dic = []
    with tqdm(total=len_iter) as progress_bar:      
        for k, batch in enumerate(data_loader):                      
            
            input_ids = batch['input_ids'].to(device)  # [batch_size, c_len]
            attn_mask = batch['attention_mask'].to(device)  # [batch_size, c_len]
            y1s = batch['token_starts'].to(device)  # [batch_size]
            y2s = batch['token_ends'].to(device)  # [batch_size]
            
            outputs = model(input_ids, attention_mask = attn_mask, 
                            start_positions = y1s, end_positions = y2s)     
            
            loss = outputs[0]
            p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]
            scores['loss'] += loss.item() 
            
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (k+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                            
            # Get start/end idxs
            p1s = utils.masked_softmax(p1s, attn_mask, dim=1, log_softmax=True)  # [batch_size, c_len]
            p2s = utils.masked_softmax(p2s, attn_mask, dim=1, log_softmax=True)  # [batch_size, c_len]
            p1s, p2s = p1s.exp(), p1s.exp()
            
            # Each record has [num_answer] candidates 
            s_idxs, e_idxs, top_probs = utils.get_ans_list_idx(p1s, p2s, num_answer=num_answer)  # [batch_size, num_answer]
            
            for i in range(p1s.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                
                if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                
                    ## True answer tokens
                    ans_piece_tokens = all_tokens[y1s[i]: (y2s[i]+1)]
                    answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                    answer = tokenizer.decode(answer_ids)
                    ans_tokens_true = answer.lower().split()
                    
                    # Convert pred idxs to tokens for each record
                    record_preds_cand = []
                    for j in range(num_answer):  # iter candidates                      
                        ans_jth_tokens = all_tokens[s_idxs[i,j]: (e_idxs[i,j]+1)]  # Token list of one answer candidate
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_jth_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_pred = answer.lower().split()  
                        record_preds_cand.append(ans_tokens_pred)  # Get list of token list of [num_answer] answer candidates
                    
                    record = {'pid': batch['pids'][i],  # 1234
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
def valid_fn_list(model, data_loader, tokenizer, device, num_answer, ans_thres):
    
    scores = {'loss': 0, 'f1': 0, 'prec': 0, 'rec': 0}
    len_iter = len(data_loader)
    n_samples = 0
    
    model.eval()
    epoch_dic = []
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:      
            for _, batch in enumerate(data_loader):                      
                
                input_ids = batch['input_ids'].to(device)  # [batch_size, c_len]
                attn_mask = batch['attention_mask'].to(device)  # [batch_size, c_len]
                y1s = batch['token_starts'].to(device)  # [batch_size]
                y2s = batch['token_ends'].to(device)  # [batch_size]
                
                outputs = model(input_ids, attention_mask = attn_mask, 
                                start_positions = y1s, end_positions = y2s)    
                
                loss = outputs[0]
                p1s, p2s = outputs[1], outputs[2]  # [batch_size, c_len]  
                scores['loss'] += loss.item() 
                
                # Get start/end idxs
                mask_c = attn_mask != torch.zeros_like(attn_mask)
                p1s = utils.masked_softmax(p1s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p2s = utils.masked_softmax(p2s, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
                p1s, p2s = p1s.exp(), p1s.exp()
                
                # Each record has [num_answer] candidates 
                s_idxs, e_idxs, top_probs = utils.get_ans_list_idx(p1s, p2s, num_answer=num_answer)  # [batch_size, num_answer]
                

                for i in range(p1s.shape[0]):
                    all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                    
                    if (y1s[i] >= 0) and (y2s[i] >= 0):  # When the answer passage not truncated
                        # True answer tokens
                        ans_piece_tokens = all_tokens[y1s[i]: (y2s[i]+1)]
                        answer_ids = tokenizer.convert_tokens_to_ids(ans_piece_tokens)
                        answer = tokenizer.decode(answer_ids)
                        ans_tokens_true = answer.lower().split()
                        
                        ## Predicted answer tokens
                        # Convert pred idxs to tokens for each record
                        record_preds_cand = []
                        for j in range(num_answer):  # iter candidates                      
                            ans_jth_tokens = all_tokens[s_idxs[i,j]: (e_idxs[i,j]+1)]  # Token list of one answer candidate
                            answer_ids = tokenizer.convert_tokens_to_ids(ans_jth_tokens)
                            answer = tokenizer.decode(answer_ids)
                            ans_tokens_pred = answer.lower().split()  
                            record_preds_cand.append(ans_tokens_pred)  # Get list of token list of [num_answer] answer candidates
                        
                        record = {'pid': batch['pids'][i],  # 1234
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