#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:23:05 2020
Ref: https://github.com/minggg/squad/blob/master/util.py

@author: qwang
"""

import torch
import torch.nn.functional as F

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
def metric_em(pred_tokens, true_tokens):    
    
    if len(pred_tokens) == len(true_tokens):
        count = 0
        for i, (true, pred) in enumerate(zip(true_tokens, pred_tokens)):
            count = (true == pred)
            count += count
        if count == len(pred_tokens):
            em = 1  
    else: 
        em = 0  
    return em



def metric_f1(pred_tokens, true_tokens):

    common = Counter(true_tokens) & Counter(pred_tokens)    
    num_same = sum(common.values())
    
    if num_same != 0:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(true_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1