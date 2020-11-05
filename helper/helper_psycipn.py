#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:49:24 2020

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
                  "question": ques,  # ques_new is only used for sentences retrieval
                  "answers": [{"text": ls['answer'], "answer_start": int(ans_start)}],
                  "context": context,
                  "group": ls['group']}
        new_list.append(record)
    
    return new_list


#%%
# with open('/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-796-factoid-30102020.json') as fin:
#     dat = json.load(fin)   
    

# dat_new = add_ans_char_pos(dat, max_n_sent=20)  # 33mins
# with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json", 'w') as fout:
#     fout.write(json.dumps(dat_new))

# dat_new = add_ans_char_pos(dat, max_n_sent=40)
# with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-40s-02112020.json", 'w') as fout:
#     fout.write(json.dumps(dat_new))
    
# dat_new = add_ans_char_pos(dat, max_n_sent=60)  # 33mins
# with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-60s-02112020.json", 'w') as fout:
#     fout.write(json.dumps(dat_new))
    
    
#%% For torchtext iterators (baseline models)
def convert_idx(text, tokens):
    """
        List of tuples for each token: (token_start_char_idx, token_end_char_idx)     
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


def process_for_baseline(dat):
    """ Obtain answer token idxs from 'answer_start'
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
        ans_text = record['answers'][0]['text'][0]
        ans_start = record['answers'][0]['answer_start']
        # Convert start/end_char_idxs to start/end_token_idxs
        if ans_start != -999:
            ans_end = ans_start + len(ans_text) - 1            
            ans_token_idxs = []
            for token_idx, span in enumerate(spans):
                if span[0] < ans_end and span[1] > ans_start:
                    ans_token_idxs.append(token_idx)
            y1s, y2s = ans_token_idxs[0], ans_token_idxs[-1]    
        else:
            y1s, y2s = -999, -999
                
        
        res_dict = {"context_tokens": context_tokens,
                    "ques_tokens": ques_tokens,
                    "answer": ans_text,
                    "y1s": y1s,
                    "y2s": y2s,
                    "id": count,
                    "group": record['group']}
        count += 1
        records.append(res_dict)
    
    return records

#%%
# # factoid-20sents
# data_path = "/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json"  
# with open(data_path) as fin:
#     dat = json.load(fin)

# dat_train, dat_valid, dat_test = [], [], []
# for ls in dat:
#     if ls['group'] == 'train':
#         dat_train.append(ls)
#     elif ls['group'] == 'valid':
#         dat_valid.append(ls)
#     else:
#         dat_test.append(ls)

# train_processed = process_for_baseline(dat_train)
# valid_processed = process_for_baseline(dat_valid)
# test_processed = process_for_baseline(dat_test)    
    
# with open(os.path.join(os.path.dirname(data_path), 'train.json'), 'w') as fout:
#     fout.write(json.dumps(train_processed))
# with open(os.path.join(os.path.dirname(data_path), 'valid.json'), 'w') as fout:
#     fout.write(json.dumps(valid_processed))        
# with open(os.path.join(os.path.dirname(data_path), 'test.json'), 'w') as fout:
#     fout.write(json.dumps(test_processed)) 