#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:26:50 2020

@author: qwang
"""

import json
import re
# from helper.text_helper import text2tokens

import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

from gensim.summarization.bm25 import BM25
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=False)

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_sci_sm")

def text2tokens(text):
    tokens = [token.text for token in nlp(text)]
    return tokens
    
def spacy_tokenizer(text):
    tokens = [token.text for token in nlp(text) if not token.is_punct]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

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
        
        
        ###### bigram-tfidf similarity ######
        if self.method == 'tfidf':
            sents_ques = sents + [ques_new]
            # vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=spacy_tokenizer, min_df=1)
            vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=spacy_tokenizer, min_df=1)
            # vectorizer = CountVectorizer(ngram_range=(2,2), tokenizer=spacy_tokenizer, min_df=1)
            sents_ques_vec = tfidf_transformer.fit_transform(vectorizer.fit_transform(sents_ques)).toarray()
            
            sents_vec = torch.from_numpy(sents_ques_vec[:-1]).float()
            ques_vec = torch.from_numpy(sents_ques_vec[-1]).float()
            cosine_scores = util.pytorch_cos_sim(sents_vec, ques_vec)
            
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
def compute_ave_precision(ans_ls, sent_ls):
    '''Compute Average Precision for a single record'''
        
    n_sent_retrieved = 0
    
    prec1, n1_relev = 0, 0  # strict
    prec2, n2_relev = 0, 0  # ratio
    prec3, n3_relev = 0, 0  # loose

    for sent in sent_ls:              
        n_sent_retrieved += 1
        
        # Check number of answers matched in "sent"
        n_ans_matched = 0
        for ans in ans_ls:
            # if ans in sent:
            matches = re.findall(r'\b'+re.escape(ans)+r'\b', sent, re.MULTILINE | re.IGNORECASE)
            if len(matches) > 0:         
                n_ans_matched += 1
                     
        # "strict": sent is relevant only when all answers can be found in the sentence
        if n_ans_matched == len(ans_ls):
            n1_relev += 1
            prec1 += n1_relev / n_sent_retrieved                               
        
        if n_ans_matched > 0:
            # "ratio": rather than 0-1, we use the ratio of matched answers as the approximate value
            n2_relev += n_ans_matched / len(ans_ls)
            prec2 += n2_relev / n_sent_retrieved
            
            # "loose": a sent is relevant if it contains at least one answer
            n3_relev += 1
            prec3 += n3_relev / n_sent_retrieved                        

    ave1_prec = prec1 / n1_relev if n1_relev > 0 else 0  # strict
    ave2_prec = prec2 / n2_relev if n2_relev > 0 else 0  # ratio
    ave3_prec = prec3 / n3_relev if n3_relev > 0 else 0  # loose 
    
    return ave1_prec, ave2_prec, ave3_prec


def compute_match_ratio(ans_ls, sent_ls, strict=True):
    '''Compute answers matched ratio for a single record'''
    
    mr1, mr2 = 0, 0
    n_ans_match = 0
    text = (" ").join(sent_ls)
    
    # Check number of answers matched in "sent"
    for ans in ans_ls:
        # if ans in text:
        # matches = re.findall(r'\b'+re.escape(ans)+r'\b', text, re.MULTILINE)
        matches = re.findall(r'\b'+re.escape(ans)+r'\b', text, re.MULTILINE | re.IGNORECASE)
        if len(matches) > 0:
            n_ans_match += 1

    mr1 = 1 if n_ans_match == len(ans_ls) else 0  # strict
    mr2 = n_ans_match / len(ans_ls)
    
    return mr1, mr2

#%%
json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-1225-30102020.json'
# json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-796-factoid-30102020.json'
# json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-429-list-30102020.json'

import time
start = time.time()
PC = PsyCIPNDataset(json_path, max_n_sent=10, method='tfidf')
sMAP, rMAP, lMAP, sMMR, MMR = 0, 0, 0, 0, 0
for i in range(len(PC)):
    ans_ls, sent_ls = PC[i][0], PC[i][1]
    
    AP = compute_ave_precision(ans_ls, sent_ls)
    sMAP += AP[0]
    rMAP += AP[1]
    lMAP += AP[2]
    
    MR = compute_match_ratio(ans_ls, sent_ls)
    sMMR += MR[0]
    MMR += MR[1]
    print(i)
    
print("sMAP|rMAP|lMAP|sMMR|MMR: |{0:.2f}|{1:.2f}|{2:.2f}|{3:.2f}|{4:.2f}".format(
    sMAP/len(PC)*100, rMAP/len(PC)*100, lMAP/len(PC)*100, sMMR/len(PC)*100, MMR/len(PC)*100))
print("Time elapsed: {} mins".format((time.time()-start)/60))  

#%%
# json_path = '/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-796-factoid-30102020.json'
# print("============ sbert ============")  
# PC = PsyCIPNDataset(json_path, max_n_sent=25, method='sbert')
# mAPs, mAP, mMRs, mMR = 0, 0, 0, 0
# for i in range(len(PC)):
#     ans_ls, sent_ls = PC[i][0], PC[i][1]
#     mAPs += compute_ave_precision(ans_ls, sent_ls, strict=True) 
#     mMRs += compute_match_ratio(ans_ls, sent_ls, strict=True)
# print("[sMAP|sMMR]-sbert-25: |{0:.2f}|{1:.2f}".format(mAPs/len(PC)*100, mMRs/len(PC)*100))