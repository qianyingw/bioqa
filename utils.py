#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:23:05 2020
Ref: https://github.com/minggg/squad/blob/master/util.py

@author: qwang
"""

import torch
import torch.nn.functional as F

import os
import shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

#%%
def masked_softmax(p, mask, dim=-1, log_softmax=False):
    """
    Take the softmax of `p` over given dimension, and set entries to 0 wherever `mask` is 0.
    Args:
        p (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `p`, with 0 indicating positions that should be assigned 0 probability in the output.
        log_softmax: Take log-softmax rather than regular softmax because `F.nll_loss` expect log-softmax.

    Returns:
        p (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    # If mask = 0, masked_p = 0 - 1e30 (~=-inf)
    # If mask = 1, masked_p = p
    masked_p = mask * p + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    p = softmax_fn(masked_p, dim)

    return p

#%%
def get_ans_idx(p_s, p_e, max_len=15, no_answer=False):
    """
    Discretize soft predictions (probs) to get start and end indices
    Choose (i, j) which maximizes p1[i]*p2[j], s.t. (i <= j) & (j-i+1 <= max_len)
    Args:
        p_s: [batch_size, c_len], probs for start index
        p_e: [batch_size, c_len], probs for end index
        max_len: max length of the answer prediction
        no_answer (bool): Treat 0-idx as the no-answer prediction. Consider a prediction no-answer
                          if preds[0,0]*preds[0,1] > the prob assigned to the max-probability span
    Returns:
        s_idxs:  [batch_size], hard predictions for start index
        e_idxs: [batch_size], hard predictions for end index
        
    """
    c_len = p_s.shape[1]
    device = p_s.device
    
    if p_s.min() < 0 or p_s.max() > 1 or p_e.min() < 0 or p_e.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probs
    p_s = p_s.unsqueeze(2)  # [batch_size, c_len, 1]
    p_e = p_e.unsqueeze(1)  # [batch_size, 1, c_len]    
    p_join = torch.bmm(p_s, p_e)  # [batch_size, c_len, c_len]

    # Restrict (i, j) s.t. (i <= j) & (j-i+1 <= max_len)
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair = is_legal_pair - torch.triu(torch.ones((c_len, c_len), device=device), diagonal=max_len)
    if no_answer:
        p_no_answer = p_join[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_join = p_join * is_legal_pair

    # Obtain (i, j) which maximizes p_join
    max_each_row, _ = torch.max(p_join, dim=2)  # [batch_size, c_len]
    max_each_col, _ = torch.max(p_join, dim=1)  # [batch_size, c_len]
    s_idxs = torch.argmax(max_each_row, dim=1)  # [batch_size]
    e_idxs = torch.argmax(max_each_col, dim=1)  # [batch_size]

    # Predict no-answer whenever p_no_answer > max_prob
    if no_answer:      
        max_prob, _ = torch.max(max_each_col, dim=1)
        s_idxs[p_no_answer > max_prob] = 0
        e_idxs[p_no_answer > max_prob] = 0

    return s_idxs, e_idxs

#%%
def get_ans_list_idx(p_s, p_e, max_len=15, threshold=None, num_answer=1):
    """
    Discretize soft predictions (probs) to get start and end indices
    Choose top [num_answer] pairs of (i, j) where p1[i]*p2[j] > threshold, s.t. (i <= j) & (j-i+1 <= max_len)
    Args:
        p_s: [batch_size, c_len], probs for start index
        p_e: [batch_size, c_len], probs for end index
        max_len: max length of the answer prediction
        num_answers: number of candidate answers returned 
        
    Returns:
        s_idxs, e_idxs (batch answer idxs)
        When threshold is None 
            -->> s_idxs, e_idxs, top_probs: tensors, [batch_size, num_answer]
        Else 
            -->> s_idxs, e_idxs: lists, element is list of ans idxs   
    """
    c_len = p_s.shape[1]
    device = p_s.device
    
    if p_s.min() < 0 or p_s.max() > 1 or p_e.min() < 0 or p_e.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probs
    p_s = p_s.unsqueeze(2)  # [batch_size, c_len, 1]
    p_e = p_e.unsqueeze(1)  # [batch_size, 1, c_len]    
    p_join = torch.bmm(p_s, p_e)  # [batch_size, c_len, c_len]
    
    # Restrict (i, j) s.t. (i <= j) & (j-i+1 <= max_len)
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair = is_legal_pair - torch.triu(torch.ones((c_len, c_len), device=device), diagonal=max_len)
    p_join = p_join * is_legal_pair
        
    if threshold:  
        batch_s_idxs, batch_e_idxs = [], []
        for p in p_join:            
            is_larger = p > threshold  # p: [c_len, c_len]
            p = p * is_larger
        
            if num_answer > 1 and torch.sum(is_larger) >= num_answer:
                # Obtain top [num_answer] (i, j) pairs where p1[i]*p2[j] > threshold
                _, flat_idxs = torch.topk(torch.flatten(p, start_dim=0), num_answer)  # [num_answer]
                idxs = torch.tensor(np.unravel_index(flat_idxs.numpy(), p.shape))  # [2, num_answer]
                s_idxs, e_idxs = idxs[0], idxs[1]            
            else:
                # Obtain (i, j) pairs where p1[i]*p2[j] > threshold
                # len(s_idxs) = 0 when torch.sum(is_larger) = 0
                s_idxs, e_idxs = torch.where(is_larger==True)  # List of answer start/end idxs for one record
                
            batch_s_idxs.append(s_idxs.tolist()) 
            batch_e_idxs.append(e_idxs.tolist())  
            
        return batch_s_idxs, batch_e_idxs  # list: element is list of ans idxs
       
    else:
        # Obtain top [num_answer] (i, j) pairs which maximize p_join
        top_probs, flat_idxs = torch.topk(torch.flatten(p_join, start_dim=1), num_answer)  # [batch_size, num_answer]
        
        s_idxs, e_idxs = [], []
        for fidx in flat_idxs:
            idxs = torch.tensor(np.unravel_index(fidx.numpy(), p_join.shape[1:]))  # [2, num_answer]
            s_idxs.append(idxs[0])  # idxs[0]: [num_answer]
            e_idxs.append(idxs[1])  # idxs[1]: [num_answer]
            
        s_idxs = torch.stack(s_idxs)    
        e_idxs = torch.stack(e_idxs)            
        return s_idxs, e_idxs, torch.sqrt(top_probs)  # tensor, [batch_size, num_answer]


#%%
def metric_em(pred_tokens, true_tokens):    
    em = 0
    if len(pred_tokens) == len(true_tokens):
        count = 0
        for i, (true, pred) in enumerate(zip(true_tokens, pred_tokens)):
            count += (true == pred)
        if count == len(pred_tokens):
            em = 1  
    return em


def metric_f1_pr(pred_tokens, true_tokens):
    """
        Calculate F1, Precision, Recall
    """
    common = Counter(true_tokens) & Counter(pred_tokens)    
    num_same = sum(common.values())  
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(true_tokens)
    if num_same != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1, precision, recall



def metric_ave_fpr(pred_token_list, true_token_list):
    """
    Compute mean f1/precision/recall for list-type questions (one record with multiple answers)
        pred_token_list: [["ab", "ef"], ["cd"]]
        true_token_list: [["ab", "ef"], ["cd"], ["fg", "b"], ...]
        
    """
    plist, rlist = [], []
    for pred_tokens in pred_token_list:
        for true_tokens in true_token_list:
            _, precision, recall = metric_f1_pr(pred_tokens, true_tokens)
            plist.append(precision)
            rlist.append(recall)
    
    ave_p = sum(plist) / len(plist)
    ave_r = sum(rlist) / len(rlist)
    if ave_p + ave_r != 0:
        ave_f1 = (2 * ave_p * ave_r) / (ave_p + ave_r)
    else:
        ave_f1 = 0
        
    return ave_f1, ave_p, ave_r



def metric_rr(pred_tokens_list, true_tokens):
    """
        Calculate Reciprocal Rank
        pred_tokens_list: list of pred_tokens candidates
        Example:
            pred_tokens_list = [['catten'], ['black', 'cat'], ['cats']]
            true_tokens = ['black', 'cat']
            RR = 1/2
    """
    rank, reciprocal_rank = 0, 0
    candidate_count = 0
    for pred_tokens in pred_tokens_list:
        candidate_count += 1
        if pred_tokens == true_tokens:
            rank = candidate_count
            break
        
    if rank != 0:      
        reciprocal_rank = 1 / rank      
    return reciprocal_rank

#%%
def ans_idx_to_tokens(context_idxs, s_idx, e_idx, BaseIter):
    """
    For baseline models
    Convert "answer idx of context" to "answer tokens" for a single record
    Input:
        context_idxs: [c_len], tensor containing context idxs of a single record
        s_idx: int, idx of answer start
        e_idx: int, idx of answer end
        BaseIter: torchtext iterators
    Output:
        ans_tokens: list of answer tokens of the single record
    """   
    
    ans_tokens = []   
    if s_idx == e_idx:       
        ans_vocab_idx = context_idxs[s_idx]  # "idx of content" =>>> "idx of TEXT vocab"
        text = BaseIter.TEXT.vocab.itos[ans_vocab_idx]  # "idx of TEXT vocab" =>>> "answer text token"
        ans_tokens.append(text.lower())
    else:
        ans_vocab_idx = context_idxs[s_idx:(e_idx+1)] 
        # ans_vocab_idx = context_idxs[s_idx:e_idx]  
        for vocab_idx in ans_vocab_idx:
            text = BaseIter.TEXT.vocab.itos[vocab_idx]
            ans_tokens.append(text.lower())
            
    return ans_tokens


#%% Checkpoint 
def save_checkpoint(state, is_best, checkdir):
    """
    Save model and training parameters at checkpoint + 'last.pth.tar'. 
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    Params:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkdir: (string) folder where parameters are to be saved
    """        
    filepath = os.path.join(checkdir, 'last.pth.tar')
    if os.path.exists(checkdir) == False:
        os.mkdir(checkdir)
    torch.save(state, filepath)    
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkdir, 'best.pth.tar'))
        
        
        
def load_checkpoint(checkfile, model, optimizer=None):
    """
    Load model parameters (state_dict) from checkfile. 
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    Params:
        checkfile: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """        
    if os.path.exists(checkfile) == False:
        raise("File doesn't exist {}".format(checkfile))
    checkfile = torch.load(checkfile)
    model.load_state_dict(checkfile['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkfile['optim_dict'])
    
    return checkfile


def save_dict_to_json(d, json_path):
    """
    Save dict of floats to json file
    d: dict of float-castable values (np.float, int, float, etc.)
      
    """      
    with open(json_path, 'w') as fout:
        d = {key: float(value) for key, value in d.items()}
        json.dump(d, fout, indent=4)
        
#%%
def plot_prfs(prfs_json_path):
    
    with open(prfs_json_path) as f:
        dat = json.load(f)
 
    # Create scores dataframe
    epochs = int(len(dat['prfs'])/2)
    train_df = pd.DataFrame(columns=['Loss', 'EM', 'F1', 'Precision', 'Recall'])
    valid_df = pd.DataFrame(columns=['Loss', 'EM', 'F1', 'Precision', 'Recall'])
    for i in range(epochs):
        train_df.loc[i] = list(dat['prfs']['train_'+str(i+1)].values())
        valid_df.loc[i] = list(dat['prfs']['valid_'+str(i+1)].values()) 
    
    # Plot
    plt.figure(figsize=(15,5))
    x = np.arange(len(train_df)) + 1   
    # Loss / F1
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(x, train_df['Loss'], label="train_loss", color='C5')
    plt.plot(x, valid_df['Loss'], label="val_loss", color='C5', linestyle='--')
    plt.xticks(np.arange(2, len(x)+2, step=2))
    plt.legend(loc='upper right')
    # Accuracy / Recall
    plt.subplot(1, 2, 2)
    plt.title("EM & F1")
    plt.plot(x, train_df['EM'], label="train_em", color='C0', alpha=0.8)
    plt.plot(x, valid_df['EM'], label="val_em", color='C0', linestyle='--', alpha=0.8)
    plt.plot(x, train_df['F1'], label="train_f1", color='C1')
    plt.plot(x, valid_df['F1'], label="val_f1", color='C1', linestyle='--')
    plt.xticks(np.arange(2, len(x)+2, step=2))
    plt.legend(loc='lower right')    