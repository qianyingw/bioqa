#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:07:13 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class Highway(nn.module):
    
    def __init__(self, hidden_dim, num_layers):
        
        super(Highway, self).__init__()
        
        self.transform_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gate_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forword(self, x):
        """
            x: [batch_size, seq_len, hidden_dim]                    
        """  
        for transform_fc, gate_fc in zip(self.transform_fcs, self.gate_fcs):
            H = F.relu(transform_fc(x))
            T = torch.sigmoid(gate_fc(x))
            x = H * T + x * (1-T)
            
        return x
            


class AttnFlow(nn.module):
    
    def __init__(self, hidden_dim, dropout):
        
        super(AttnFlow, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
    
    def get_similarity_matrix(self, c, q):
    
    def forward(self, c, q):
    
    
    
    self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))


class BiDAF(nn.module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):      
        
        super(BiDAF, self).__init__()     
    
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.embed_fc = nn.Linear(embed_dim, hidden_dim, bias = False)
        self.highway = Highway(hidden_dim, 2)
        
        self.lstm_encoder = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = num_layers,
                                    batch_first = True, bidirectional = True,
                                    dropout = dropout if num_layers > 1 else 0)        
        self.dropout = nn.Dropout(dropout)
        
        # C2Q attention
        # Q2C attention

