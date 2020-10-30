#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:49:24 2020

@author: qwang
"""


import json
import re

import torch

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=False)

from sentence_transformers import util

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
def add_ans_char_pos(data_list, max_n_sent):
    ''' 
    Extract most [max_n_sent] similar sentences and convert to new "context"
    Save char positions of first exact match
    Generate new list include ans char position
    '''
    new_list = []
    
    for ls in data_list:         
        QuesID = ls['QuesID']
        sents = ls['sentences']            
        if QuesID.split('_')[1] == 'a':
            ques = 'What is the method of induction of disease model?'
        else:
            ques = 'What is the intervention?'
              
        ## Extract most similar sents by bigram-tfidf
        ques_new = ques + ' ' + ls['title']
        sents_ques = sents + [ques_new]
        vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=spacy_tokenizer, min_df=1)
        sents_ques_vec = tfidf_transformer.fit_transform(vectorizer.fit_transform(sents_ques)).toarray()
        
        sents_vec = torch.from_numpy(sents_ques_vec[:-1]).float()
        ques_vec = torch.from_numpy(sents_ques_vec[-1]).float()
        cosine_scores = util.pytorch_cos_sim(sents_vec, ques_vec)
        
        pairs = []
        for i in range(cosine_scores.shape[0]):
            pairs.append({'index': i, 'score': cosine_scores[i][0]})
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
                 
        sents_new = []
        if len(sents) > max_n_sent:
            for pair in pairs[:max_n_sent]:
                sents_new.append(sents[pair['index']])
        else:
            for pair in pairs:
                sents_new.append(sents[pair['index']])
        
        # Convert 'new' sents back to text
        context = (' ').join(sents_new)
        
        # Find char pos of first answer match        
        regex = r'\b' + re.escape(ls['answer'][0]) + r'\b'
        matches = re.finditer(regex, context, re.MULTILINE | re.IGNORECASE)
        n_matches = 0 
        for matchId, match in enumerate(matches, start=1):
            n_matches += 1
            if n_matches == 1:
                ans_start = match.start()
                break
        if n_matches == 0:
            ans_start = -999
        
        record = {"QuesID": ls["QuesID"],
                  "PubID": ls['PubID'],
                  "title": ls['title'],
                  "question": ques,
                  "answer": ls['answer'][0],
                  "answer_start": ans_start,
                  "context": context,
                  "group": ls['group']}
        new_list.append(record)
    
    return new_list



    
