#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:26:50 2020

@author: qwang
"""

import json
import re
# from helper.text_helper import text2tokens
import spacy
nlp = spacy.load("en_core_sci_sm")
def text2tokens(text):
    tokens = [token.text for token in nlp(text)]
    return tokens
    


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
        answers = self.dat[idx]['answer']
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
            if len(sents) > self.max_n_sent:
                for pair in pairs[:self.max_n_sent]:
                    sent_ls.append(sents[pair['index']])
            else:
                for pair in pairs:
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
            if len(sents) > self.max_n_sent:
                for sidx in sorted_idx[:self.max_n_sent]:
                    sent_ls.append(sents[sidx])  
            else:
                for sidx in sorted_idx:
                    sent_ls.append(sents[sidx])  
                

        return answers, sent_ls
    
        
#%%
def compute_ave_precision(ans_ls, sent_ls, strict=True):
    '''Compute Average Precision for a single record'''
        
    ave_precision = 0
    n_sent_retrieved = 0
    n_sent_relevant = 0
    
    for sent in sent_ls:              
        n_sent_retrieved += 1
        
        # Check number of answers matched in "sent"
        n_ans_matched = 0
        for ans in ans_ls:
            # if ans in sent:
            matches = re.findall(r'\b'+re.escape(ans)+r'\b', sent, re.MULTILINE)
            if len(matches) > 0:         
                n_ans_matched += 1
                
        if strict == True:       
            # "sent" is relevamt only when all answers can be found in the sentence
            if n_ans_matched == len(ans_ls):
                n_sent_relevant += 1
                precision = n_sent_relevant / n_sent_retrieved
                ave_precision += precision
        else:
            # Rather than 0-1, we use the ratio of matched answers as the approximate value
            if n_ans_matched > 0:
                n_sent_relevant += n_ans_matched / len(ans_ls)
                precision = n_sent_relevant / n_sent_retrieved
                ave_precision += precision
    
    if n_sent_relevant == 0:
        ave_precision = 0
    else:
        ave_precision = ave_precision / n_sent_relevant
              
    return ave_precision


def compute_match_ratio(ans_ls, sent_ls, strict=True):
    '''Compute answers matched ratio for a single record'''
    match_ratio = 0
    n_ans_match = 0
    text = (" ").join(sent_ls)
    
    # Check number of answers matched in "sent"
    for ans in ans_ls:
        # if ans in text:
        matches = re.findall(r'\b'+re.escape(ans)+r'\b', text, re.MULTILINE)
        if len(matches) > 0:
            n_ans_match += 1
    
    if strict == True:
        if n_ans_match == len(ans_ls):
            match_ratio = 1
    else:
        match_ratio = n_ans_match / len(ans_ls)
              
    return match_ratio

#%%
# json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-992-24102020.json'
json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-679-factoid-28102020.json'
# json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-313-28102020.json'

### SBERT 
PC = PsyCIPNDataset(json_path, max_n_sent=20, method='sbert')
mAPs, mAP = 0, 0
mMRs, mMR = 0, 0
for i in range(len(PC)):
    ans_ls, sent_ls = PC[i][0], PC[i][1]
    mAPs += compute_ave_precision(ans_ls, sent_ls, strict=True) 
    mAP += compute_ave_precision(ans_ls, sent_ls, strict=False)  
    mMRs += compute_match_ratio(ans_ls, sent_ls, strict=True)
    mMR += compute_match_ratio(ans_ls, sent_ls, strict=False)
    # print(i)
print("============ SBERT with 20 sents ============")    
print("[sMAP|MAP|sMMR|MMR]: {0:.2f}|{1:.2f}|{2:.2f}|{3:.2f}".format(mAPs/len(PC)*100, mAP/len(PC)*100, mMRs/len(PC)*100, mMR/len(PC)*100))

