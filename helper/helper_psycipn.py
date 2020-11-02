#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:49:24 2020

@author: qwang
"""


import json
import re

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
                  "question": ques,
                  "answers": [{"text": ls['answer'], "answer_start": int(ans_start)}],
                  "context": context,
                  "group": ls['group']}
        new_list.append(record)
    
    return new_list


#%%
with open('/media/mynewdrive/bioqa/PsyCIPN-InduceIntervene-796-factoid-30102020.json') as fin:
    dat = json.load(fin)   
    

dat_new = add_ans_char_pos(dat, max_n_sent=20)  # 33mins
with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-20s-02112020.json", 'w') as fout:
    fout.write(json.dumps(dat_new))

dat_new = add_ans_char_pos(dat, max_n_sent=40)
with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-40s-02112020.json", 'w') as fout:
    fout.write(json.dumps(dat_new))
    
dat_new = add_ans_char_pos(dat, max_n_sent=60)  # 33mins
with open("/media/mynewdrive/bioqa/PsyCIPN-II-796-factoid-60s-02112020.json", 'w') as fout:
    fout.write(json.dumps(dat_new))