#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:23:23 2021

@author: qwang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
 

class PositionEncoder(nn.Module):
    ''' Positional encoding
    '''
    def __init__(self, s_len, d_model):
        super().__init__()
        
        freqs, phases = [], []
        for i in range(d_model):
            if i % 2 == 0:
                freq = 10000 ** (-i / d_model)
                phase = 0
            else:
                freq = -10000 ** ((1-i) / d_model)
                phase = math.pi / 2
            freqs.append(freq)
            phases.append(phase)
            
        freqs = torch.Tensor(freqs).unsqueeze(1)  # [d_model, 1]
        phases = torch.Tensor(phases).unsqueeze(1)  # [d_model, 1]
        pos = torch.arange(s_len).repeat(d_model, 1).to(torch.float)  # [d_model, s_len]
        
        pos_enc = torch.sin(torch.add(torch.mul(pos, freqs), phases))  # [d_model, s_len]
        self.pos_enc = nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, x):
        '''
            x: [batch_size, s_len, d_model]
        '''
        x = x.permute(0,2,1)  # [batch_size, d_model, s_len]
        x = x + self.pos_enc  # [batch_size, d_model, s_len]
        x = x.permute(0,2,1)  # [batch_size, s_len, d_model]
        return x
        
#%%
class InputEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, dropout=0.1):
        super(InputEmbed, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.highway = Highway(embed_dim, 2)     

    def forward(self, idx_seq):
        '''      
            idx_seq: [batch_size, seq_len]
        '''
        seq_embed = self.embed(idx_seq)  # [batch_size, seq_len, embed_dim]
        seq_dp = self.dropout(seq_embed) 
        seq_hw = self.highway(seq_dp)
        
        return seq_hw  # [batch_size, seq_len, embed_dim]
    
    
#%%    
class SepConv(nn.Module):
    ''' Depthwise separable convolution 
    '''
    def __init__(self, hdim, odim, fsize):
        super(SepConv, self).__init__()
        # When groups==in_channels & out_channels==K*in_channels, the operation is depthwise convolution
        self.depthwise_conv = nn.Conv1d(in_channels=hdim, out_channels=hdim, kernel_size=fsize, groups=hdim, padding=fsize//2)
        self.pointwise_conv = nn.Conv1d(in_channels=hdim, out_channels=odim, kernel_size=1, padding=0)
        
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)
    
    def forward(self, seq):
        ''' seq: [batch_size, ?, hdim] 
        '''
        seq = seq.permute(0,2,1)  # [batch_size, hdim, ?]
        out = self.depthwise_conv(seq)  # [batch_size, hdim, ?+2*(fsize//2)-fsize+1] == [batch_size, hdim, ?]
        out = self.pointwise_conv(out)  # [batch_size, odim, ?-1+1] == [batch_size, odim, ?]
        out = out.permute(0,2,1)  # [batch_size, ?, hdim] 
        return out
    

# depth_conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, groups=128, padding=5//2)
# point_conv = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
# seq = torch.randn(16, 88, 128)  # [batch_size, s_len, hdim]
# depth_conv(seq.permute(0,2,1)).shape  # [16, 128, 88]  [batch_size, hidden_dim, s_len]
# point_conv(depth_conv(seq.permute(0,2,1))).shape  # [16, 64, 88]  # [batch_size, odim, s_len]


class EncoderBlock(nn.Module):
    
    def __init__(self, num_convs, s_len, hidden_dim, fsize):
        super(EncoderBlock, self).__init__()
        
        self.pos_enc = PositionEncoder(s_len, hidden_dim)   
        self.sep_convs = nn.ModuleList([SepConv(hidden_dim, hidden_dim, fsize) for _ in range(num_convs)])   
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim, s_len) for _ in range(num_convs)])                           
        self.dropout = nn.Dropout(p=0.1)

        self.enc_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=8)  # self-attn + ffn
         
    def forward(self, x, x_mask):
        '''
            x: [batch_size, s_len, hidden_dim]
            x_mask: [batch_size, s_len]
        '''
        
        out = self.pos_enc(x)  # [batch_size, s_len, hidden_dim]
        res = out
        for i, sep_conv in enumerate(self.sep_convs):
            # out = self.layer_norms[i](out)  # [batch_size, s_len, hidden_dim]
            out = sep_conv(out)  # [batch_size, s_len, hidden_dim]
            out = self.layer_norms[i](out)  # [batch_size, s_len, hidden_dim]
            out = self.dropout(F.relu(out))
            out = out + res  # update out
            res = out  # update res
            
        ## self-attn + ffn
        out = out.permute(1,0,2)  # [s_len, batch_size, hidden_dim]
        # In TransformerEncoderLayer, 
        #       src: [s_len, batch_size, hidden_dim]
        #       src_key_padding_mask: [batch_size, s_len], 1 -> pad token, 0 -> true token
        x_mask = torch.logical_not(x_mask)
        out = self.enc_layer(src=out, src_key_padding_mask=x_mask)  # [s_len, batch_size, hidden_dim]        
        out = out.permute(1,0,2)  # [batch_size, s_len, hidden_dim]
        return out  

#%%
class CQAttn(nn.Module):
    
    def __init__(self, hidden_dim, dropout=0.1):
        
        super(CQAttn, self).__init__()
        
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


#%%
class QANet(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, max_c_len, max_q_len, hidden_dim, n_block_mod, pad_idx=1):
        super(QANet, self).__init__()
        
        self.embed_inp = InputEmbed(vocab_size, embed_dim, pad_idx)
        self.embed_conv = SepConv(hdim = embed_dim, odim = hidden_dim, fsize=5)
        
        self.embed_enc_c = EncoderBlock(num_convs=4, s_len=max_c_len, hidden_dim=hidden_dim, fsize=7)
        self.embed_enc_q = EncoderBlock(num_convs=4, s_len=max_q_len, hidden_dim=hidden_dim, fsize=7)
        
        self.attn = CQAttn(hidden_dim)
        
        self.mod_conv = SepConv(hdim = 4*hidden_dim, odim = hidden_dim, fsize=5)
        mod_enc_block = EncoderBlock(num_convs=2, s_len=max_c_len, hidden_dim=hidden_dim, fsize=5)
        self.mod_enc = nn.ModuleList([mod_enc_block] * n_block_mod)
        
        self.fc1 = nn.Linear(2*hidden_dim, 1)
        self.fc2 = nn.Linear(2*hidden_dim, 1)
        

    def forward(self, idx_c, idx_q, y1s, y2s):
        """
            idx_c: [batch_size, c_len]
            idx_q: [batch_size, q_len]  
            y1s: [batch_size], start idxs of true answers
            y2s: [batch_size], end idxs of true answers  
        """ 
        mask_c = torch.zeros_like(idx_c) != idx_c
        mask_q = torch.zeros_like(idx_q) != idx_q
        
        # 1. Inpur embed layer
        emb_c = self.embed_inp(idx_c)  # [batch_size, c_len, embed_dim]
        emb_q = self.embed_inp(idx_q)  # [batch_size, q_len, embed_dim]
        # connection between embed layer and embed encoder
        emb_c = self.embed_conv(emb_c)  # [batch_size, c_len, embed_dim]
        emb_q = self.embed_conv(emb_q)  # [batch_size, q_len, embed_dim]
        
        # 2. Embed encoder
        emb_enc_c = self.embed_enc_c(emb_c, mask_c)  # [batch_size, c_len, hidden_dim]
        emb_enc_q = self.embed_enc_q(emb_q, mask_q)  # [batch_size, q_len, hidden_din]
        
        # 3. CQ attention
        G = self.attn(emb_enc_c, emb_enc_q, mask_c, mask_q)  # [batch_size, c_len, 4*hidden_dim]
        # Connection between attn and model encoder
        G = self.mod_conv(G)  # [batch_size, c_len, hidden_dim]

        # 4. Model encoders
        M0 = G
        for enc in self.mod_enc:
            M0 = enc(M0, mask_c)  # [batch_size, c_len, hidden_dim]
            
        M1 = M0
        for enc in self.mod_enc:
            M1 = enc(M1, mask_c)  # [batch_size, c_len, hidden_dim]
            
        M2 = M1
        for enc in self.mod_enc:
            M2 = enc(M2, mask_c)  # [batch_size, c_len, hidden_dim]
            
        # 5. Output      
        x1 = torch.cat([M0, M1], dim=2)  # [batch_size, c_len, 2*hidden_dim]
        x2 = torch.cat([M0, M2], dim=2)  # [batch_size, c_len, 2*hidden_dim]
        
        logits1 = self.fc1(x1).squeeze(2)  # [batch_size, c_len]
        logits2 = self.fc2(x2).squeeze(2)  # [batch_size, c_len]
             
        # log_p1 = masked_softmax(logits1, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
        # log_p2 = masked_softmax(logits2, mask_c, dim=1, log_softmax=True)  # [batch_size, c_len]
        p1 = masked_softmax(logits1, mask_c, dim=1, log_softmax=False)  # [batch_size, c_len]
        p2 = masked_softmax(logits2, mask_c, dim=1, log_softmax=False)  # [batch_size, c_len]
        
        ### Loss ###     
        # Sometimes y1s/y2s are outside the model inputs (like -999), need to ignore these terms
        ignored_idx = p1.shape[1]
        y1s_clamp = torch.clamp(y1s, min=0, max=ignored_idx)  # limit value to [0, max_c_len]. '-999' converted to 0 
        y2s_clamp = torch.clamp(y2s, min=0, max=ignored_idx)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_idx)
        loss = (loss_fn(logits1, y1s_clamp) + loss_fn(logits2, y2s_clamp)) / 2 

        return loss, p1, p2
        