#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:07:13 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_softmax

#%%
class Highway(nn.Module):
    
    def __init__(self, hidden_dim, num_layers):
        
        super(Highway, self).__init__()       
        self.transform_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gate_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        """
            x: [batch_size, seq_len, hidden_dim]                    
        """  
        for transform_fc, gate_fc in zip(self.transform_fcs, self.gate_fcs):
            H = F.relu(transform_fc(x))
            T = torch.sigmoid(gate_fc(x))
            x = H * T + x * (1-T)
            
        return x
            


class AttnFlow(nn.Module):
    
    def __init__(self, hidden_dim, dropout):
        
        super(AttnFlow, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weight
        self.W_c = nn.Parameter(torch.empty(hidden_dim, 1))
        self.W_q = nn.Parameter(torch.empty(hidden_dim, 1))
        self.W_cq = nn.Parameter(torch.empty(1, 1, hidden_dim))
        for w in (self.W_c, self.W_q, self.W_cq):
            nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        
        # Initialize bias
        self.b = nn.Parameter(torch.zeros(1))  
        
    def forward(self, c, q, mask_c, mask_q):
        """
            c: [batch_size, c_len, hidden_dim]
            q: [batch_size, q_len, hidden_dim]  
            mask_c: [batch_size, c_len]
            mask_q: [batch_size, q_len]  
        """ 
        
        S = self.get_similarity_matrix(c, q)  # [batch_size, c_len, q_len]
            
        # S1 = F.softmax(S, dim=2)  # row-wise softmax. [batch_size, c_len, q_len]
        # S2 = F.softmax(S, dim=1)  # column-wise softmax. [batch_size, c_len, q_len]
        
        mask_c = mask_c.unsqueeze(2)  # [batch_size, c_len, 1]
        mask_q = mask_q.unsqueeze(1)  # [batch_size, 1, q_len]
        S1 = masked_softmax(S, mask_q, dim=2)  # row-wise softmax. [batch_size, c_len, q_len]
        S2 = masked_softmax(S, mask_c, dim=1)  # column-wise softmax. [batch_size, c_len, q_len]
        
        # C2Q attention
        A = torch.bmm(S1, q)  # [batch_size, c_len, hidden_dim]
        
        # Q2C attention
        S_temp = torch.bmm(S1, S2.transpose(1,2))  # [batch_size, c_len, c_len]
        B = torch.bmm(S_temp, c)    # [batch_size, c_len, hidden_dim]
              
        # c: [batch_size, c_len, hidden_dim]
        # A: [batch_size, c_len, hidden_dim]
        # torch.mul(c, A): [batch_size, c_len, hidden_dim]
        # torch.mul(c, B): [batch_size, c_len, hidden_dim]
        G = torch.cat((c, A, torch.mul(c, A), torch.mul(c, B)), dim=2)  # [batch_size, c_len, 4*hidden_dim]
        
        return G
    
    def get_similarity_matrix(self, c, q):
        """
            c: [batch_size, context_len, hidden_dim]
            q: [batch_size, question_len, hidden_dim]  
            W_c: [hidden_dim, 1]
            W_q: [hidden_dim, 1]
            W_cq: [1, 1, hidden_dim]
        """ 

        c_len = c.shape[1]        
        q_len = q.shape[1]
    
        c = self.dropout(c)  
        q = self.dropout(q)

        S_c = torch.matmul(c, self.W_c)  # [batch_size, c_len, 1]
        S_c = S_c.expand([-1, -1, q_len])  # [batch_size, c_len, q_len]
        
        S_q = torch.matmul(q, self.W_q).transpose(1, 2)  # [batch_size, 1, q_len]
        S_q = S_q.expand([-1, c_len, -1])  # [batch_size, c_len, q_len]
        
        # matmul([batch_size, c_len, hidden_dim], [batch_size, hidden_dim, q_len]) --> [batch_size, c_len, q_len]
        S_cq = torch.matmul(c * self.W_cq, q.transpose(1, 2))  

        S = S_c + S_q + S_cq + self.b
        
        return S
    

class BiDAF(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):      
        
        super(BiDAF, self).__init__()     
    
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.embed_fc = nn.Linear(embed_dim, hidden_dim, bias = False)
        self.highway = Highway(hidden_dim, 2)
        
        self.enc = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = num_layers,
                           batch_first = True, bidirectional = True,
                           dropout = dropout if num_layers > 1 else 0)        
        self.dropout = nn.Dropout(dropout)
        
        # Attention flow
        self.attn_flow = AttnFlow(2*hidden_dim, dropout)
        
        # Modelling
        self.mod = nn.LSTM(input_size = 8*hidden_dim, hidden_size = hidden_dim, num_layers = 2,
                           batch_first = True, bidirectional = True,
                           dropout = dropout)         
        # Output
        # Start pointer
        self.attn_fc1 = nn.Linear(8*hidden_dim, 1)
        self.mod_fc1 = nn.Linear(2*hidden_dim, 1)
        
        # End pointer
        self.enc_M = nn.LSTM(input_size = 2*hidden_dim, hidden_size = hidden_dim, num_layers = 1,
                             batch_first = True, bidirectional = True,
                             dropout = 0)        
        self.attn_fc2 = nn.Linear(8*hidden_dim, 1)
        self.mod_fc2 = nn.Linear(2*hidden_dim, 1)
        
    
    def forward(self, idx_c, idx_q, y1s, y2s):
        """
            idx_c: [batch_size, context_len]
            idx_q: [batch_size, question_len]  
            y1s: [batch_size], start idxs of true answers
            y2s: [batch_size], end idxs of true answers
            
        """ 
        mask_c = torch.zeros_like(idx_c) != idx_c
        mask_q = torch.zeros_like(idx_q) != idx_q
              
        embed_c = self.embedding(idx_c)  # [batch_size, c_len, embed_dim]
        embed_q = self.embedding(idx_q)  # [batch_size, q_len, embed_dim]
        
        embed_fc_c = self.embed_fc(embed_c)  # [batch_size, c_len, hidden_dim]
        embed_fc_q = self.embed_fc(embed_q)  # [batch_size, q_len, hidden_dim]
        
        highway_c = self.highway(embed_fc_c)  # [batch_size, c_len, hidden_dim]
        highway_q = self.highway(embed_fc_q)  # [batch_size, q_len, hidden_dim]
                
        enc_c, _ = self.enc(highway_c)  # [batch_size, c_len, 2*hidden_dim]     
        enc_c = self.dropout(enc_c) 
        
        enc_q, _ = self.enc(highway_q)  # [batch_size, q_len, 2*hidden_dim]
        enc_q = self.dropout(enc_q)       

        # Attention flow layer
        G = self.attn_flow(enc_c, enc_q, mask_c, mask_q)  # [batch_size, c_len, 8*hidden_dim]
        
        # Modeling layer
        M, _ = self.mod(G)  # [batch_size, c_len, 2*hidden_dim]
        M_new, _ = self.enc_M(M)  # [batch_size, c_len, 2*hidden_dim]
        
        # Output
        logits_1 = self.attn_fc1(G) + self.mod_fc1(M)  # [batch_size, c_len, 1]
        logits_2 = self.attn_fc2(G) + self.mod_fc2(M_new)  # [batch_size, c_len, 1]
        logits_1 = logits_1.squeeze(2)  # [batch_size, c_len]      
        logits_2 = logits_2.squeeze(2)  # [batch_size, c_len] 
             
        # log_p1 = masked_softmax(logits_1, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
        # log_p2 = masked_softmax(logits_2, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
        p1 = masked_softmax(logits_1, mask_c, dim=1, log_softmax=False)  # [batch_size, c_len]
        p2 = masked_softmax(logits_2, mask_c, dim=1, log_softmax=False)  # [batch_size, c_len]
        
        ### Loss ###     
        # Sometimes y1s/y2s are outside the model inputs (like -999), need to ignore these terms
        ignored_idx = p1.shape[1]
        y1s_clamp = torch.clamp(y1s, min=0, max=ignored_idx)  # limit value to [0, max_c_len]. '-999' converted to 0 
        y2s_clamp = torch.clamp(y2s, min=0, max=ignored_idx)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_idx)
        loss = (loss_fn(logits_1, y1s_clamp) + loss_fn(logits_2, y2s_clamp)) / 2         
        
        return loss, p1, p2