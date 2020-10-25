#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:42:18 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_softmax

#%%
hidden_dim = 128
num_layers = 2
batch_size = 32
c_len = 666
q_len = 7
dp_rate = 0.5
vocab_size = 1000000
embed_dim = 200


x = torch.rand((batch_size, c_len, hidden_dim))



x = torch.randint(0, 9999, (batch_size, c_len, hidden_dim))
x.type()

#%% Check Highway
transform_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
gate_fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

### Forward
for transform_fc, gate_fc in zip(transform_fcs, gate_fcs):
    H = F.relu(transform_fc(x))
    T = torch.sigmoid(gate_fc(x))
    x = H * T + x * (1-T)
    print(x[1,1,1])

print(H.shape)
print(T.shape)
print(x.shape)

#%% Check AttnFlow
dropout = nn.Dropout(dp_rate)
W_c = nn.Parameter(torch.empty(hidden_dim, 1))
W_q = nn.Parameter(torch.empty(hidden_dim, 1))
W_cq = nn.Parameter(torch.empty(1, 1, hidden_dim))
for w in (W_c, W_q, W_cq):
    nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
b = nn.Parameter(torch.zeros(1))  
        

### Similarity matrix
c = torch.rand((batch_size, c_len, hidden_dim))
q = torch.rand((batch_size, q_len, hidden_dim))

c = dropout(c)  
q = dropout(q)

S_c = torch.matmul(c, W_c)  # [batch_size, c_len, 1]
S_c = S_c.expand([-1, -1, q_len])  # [batch_size, c_len, q_len]

S_q = torch.matmul(q, W_q).transpose(1, 2)  # [batch_size, 1, q_len]
S_q = S_q.expand([-1, c_len, -1])  # [batch_size, c_len, q_len]

S_cq = torch.matmul(c * W_cq, q.transpose(1, 2))   # matmul([batch_size, c_len, hidden_dim], [batch_size, hidden_dim, q_len])
S = S_c + S_q + S_cq + b  # [batch_size, c_len, q_len]

print(S.shape)


### Forward
idx_c = torch.rand((batch_size, c_len))
idx_q = torch.rand((batch_size, q_len))

mask_c = torch.zeros_like(idx_c) != idx_c
mask_q = torch.zeros_like(idx_q) != idx_q
        
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
        

        
#%% Check BiDAF
from model import Highway, AttnFlow

embedding = nn.Embedding(vocab_size, embed_dim, pad_idx=1)
embed_fc = nn.Linear(embed_dim, hidden_dim, bias = False)
highway = Highway(hidden_dim, 2)
enc = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = num_layers,
              batch_first = True, bidirectional = True,
              dropout = dropout if num_layers > 1 else 0)     
   
dropout = nn.Dropout(dropout)

# Attention flow
attn_flow = AttnFlow(2*hidden_dim, dropout)
# Modelling
mod = nn.LSTM(input_size = 8*hidden_dim, hidden_size = hidden_dim, num_layers = 2,
                   batch_first = True, bidirectional = True,
                   dropout = dropout)         
# Output
# Start pointer
attn_fc1 = nn.Linear(8*hidden_dim, 1)
mod_fc1 = nn.Linear(2*hidden_dim, 1)
# End pointer
enc_M = nn.LSTM(input_size = 2*hidden_dim, hidden_size = hidden_dim, num_layers = 1,
                     batch_first = True, bidirectional = True,
                     dropout = 0)        
attn_fc2 = nn.Linear(8*hidden_dim, 1)
mod_fc2 = nn.Linear(2*hidden_dim, 1)



### Forward
