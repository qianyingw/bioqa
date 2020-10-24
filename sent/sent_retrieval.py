#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:26:50 2020

@author: qwang
"""

import json

from helper.text_helper import text2tokens

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

from gensim.summarization.bm25 import BM25

#%%
class PsyCIPNDataset():
    
    def __init__(self, json_path, max_n_sent, method):
        
        with open(json_path) as fin:
            dat = json.load(fin)
        # if group:
        #     dat = info_df[info_df['partition']==group]
        # self.info_df = info_df.reset_index(drop=True)      
        
        self.dat = dat
        self.max_n_sent = max_n_sent
        self.method = method
    
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, idx):
        
        QuesID = self.dat[idx]['QuesID']
        title = self.dat[idx]['title']
        # answers = self.dat[idx]['answer']
        sents = self.dat[idx]['sentences']
        
        if QuesID.split('_')[1] == 'a':
            ques = 'What is the method of induction of disease model?'
        else:
            ques = 'What is the intervention?'
        
        ques_new = ques + ' ' + title

        ###### sentence-bert similarity ######
        if self.method == 'sbert':
            
            sent_embeds = model.encode(sents, convert_to_tensor=True)
            ques_embed = model.encode([ques_new], convert_to_tensor=True)
    
            # Compute cosine-similarities for each sent with other sents
            cosine_scores = util.pytorch_cos_sim(sent_embeds, ques_embed)
            
            # Find the pairs with the highest cosine similarity scores
            pairs = []
            for i in range(cosine_scores.shape[0]):
                pairs.append({'index': i, 'score': cosine_scores[i][0]})
            pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
                     
            sent_ls = []
            for pair in pairs[:self.max_n_sent]:
                sent_ls.append(sents[pair['index']])
                
        ###### bm25 ######
        if self.method == 'bm25':
            
            sent_tokens = [text2tokens(s) for s in sents]
            ques_tokens = text2tokens(ques_new)
            
            bm25 = BM25(sent_tokens)
            scores = bm25.get_scores(ques_tokens)
            scores_dict = dict(zip(range(len(scores)), scores))
            sorted_idx = sorted(scores_dict, key=scores_dict.get, reverse=True)

            sent_ls = []
            for sidx in sorted_idx[:self.max_n_sent]:
                sent_ls.append(sents[sidx])           

        return sent_ls

        
#%%
json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-1052-24102020.json'

PC = PsyCIPNDataset(json_path, max_n_sent=10, method='sbert')
PC = PsyCIPNDataset(json_path, max_n_sent=10, method='bm25')

temp = PC[0]



