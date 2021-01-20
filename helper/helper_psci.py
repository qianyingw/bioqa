#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:36:46 2021

@author: qwang
"""
import json
import re
import os

# import torch
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# tfidf_transformer = TfidfTransformer(smooth_idf=False)

from sentence_transformers import SentenceTransformer, util
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

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
    Save char positions of first exact match for each answer candidate
    Generate new list include ans char position
    '''
    k=0    
    
    new_list = []
    
    for ls in data_list:         
        QuesID = ls['QuesID']
        sents = ls['sentences']            
        if QuesID.split('_')[1] == 'a':
            ques = 'What is the method of induction of disease model?'
        else:
            ques = 'What is the intervention?'
              
        ## Extract most similar sents by sentence-bert
        ques_new = ques + ' ' + ls['title']      
        sent_embeds = sent_model.encode(sents, convert_to_tensor=True)
        ques_embed = sent_model.encode([ques_new], convert_to_tensor=True)

        # Compute cosine-similarities for each sent with other sents
        cosine_scores = util.pytorch_cos_sim(sent_embeds, ques_embed)
        
        # Find the pairs with the highest cosine similarity scores
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
        
        # Find char pos of first answer match for each answer candidate
        for ans_text in ls['answer']:           
            regex = r'\b' + re.escape(ans_text) + r'\b'
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
                      "question": ques,  # ques_new is only used for sentences retrieval
                      "answers": {"text": ans_text, "answer_start": int(ans_start)},
                      "context": context,
                      "group": ls['group']}
            
            new_list.append(record)                
                
        k = k+1
        print(k)
    return new_list

#%%
with open('/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-796-factoid-30102020.json') as fin:
    dat = json.load(fin)   
    
dat_new = add_ans_char_pos(dat, max_n_sent=20)  # 33mins
with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json", 'w') as fout:
    fout.write(json.dumps(dat_new))
    
    
# All 1225 records
with open('/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-1225-30102020.json') as fin:
    dat = json.load(fin)   

start = time.time()
dat_new = add_ans_char_pos(dat, max_n_sent=30)
print("Time elapsed: {} mins".format((time.time()-start)/60))  
  
with open("/media/mynewdrive/bioqa/PsyCIPN-II-1225-30s-20012021.json", 'w') as fout:
    fout.write(json.dumps(dat_new))

#%% For torchtext iterators (baseline models)
def convert_idx(text, tokens):
    """ Generate spans for answer tokens
        spans: List of tuples for each token: (token_start_char_idx, token_end_char_idx)     
    """
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)  # Find position of 1st occurrence; start search from 'current' 
        if current < 0:
            raise Exception(f"Token '{token}' cannot be found")
        spans.append((current, current + len(token)))
        current += len(token)  # next search start from the token afterwards
    return spans


def charIdx_to_tokenIdx(spans, ans_text, ans_start):
    """ 
        Convert answer 'char idx' to 'token idx' for one single record
    """        
    # Convert answer 'char idxs' to 'token idxs' for one single record
    if ans_start != -999:
        ans_end = ans_start + len(ans_text) - 1            
        ans_token_idxs = []
        for token_idx, span in enumerate(spans):
            if span[0] < ans_end and span[1] > ans_start:
                ans_token_idxs.append(token_idx)
        y1, y2 = ans_token_idxs[0], ans_token_idxs[-1]    
    else:
        y1, y2 = -999, -999
         
    return y1, y2


def process_for_baseline(dat):
    """ Obtain answer token idxs from 'answer_start'
        Each record in dat has one answer string 
        (Question with multiple answers should be split into separate records previously)
    """
    records = []
    count = 0
    for record in dat:
        # Context
        context = record['context'].replace("''", '" ').replace("``", '" ') # Replace non-standard quotation marks
        context_tokens = text2tokens(context)
        spans = convert_idx(context, context_tokens)  # (token_start_char_idx, token_end_char_idx)
        
        # Question
        ques = record['question'].replace("''", '" ').replace("``", '" ')  
        ques_tokens = text2tokens(ques)
        
        # Answers
        y1, y2 = charIdx_to_tokenIdx(spans, record['answers']['text'], record['answers']['answer_start'])   

        res_dict = {"context_tokens": context_tokens,
                    "ques_tokens": ques_tokens,
                    "answer": record['answers']['text'],  
                    "y1s": y1,  
                    "y2s": y2,
                    "id": count,
                    'pubId': record['PubID'],
                    "group": record['group']}
        records.append(res_dict)
        count += 1             
        
    return records