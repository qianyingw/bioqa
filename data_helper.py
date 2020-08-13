#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:47:55 2020

@author: qwang
"""

import json
import spacy
import os
nlp = spacy.load("en_core_web_sm")

#%%
def word_tokenizer(text, lower=True):
    if lower:
        tokens = [token.text.lower() for token in nlp(text)]
    else:
        tokens = [token.text for token in nlp(text)]
    return tokens

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


def process_records(dat, lower):
    
    records = []
    count = 0
    for record in dat:
        # Context
        context = record['context'].replace("''", '" ').replace("``", '" ') # Replace non-standard quotation marks
        context_tokens = word_tokenizer(context, lower)
        # (token_start_char_idx, token_end_char_idx)     
        spans = convert_idx(context, context_tokens)
        
        # Question
        ques = record['question'].replace("''", '" ').replace("``", '" ')  
        ques_tokens = word_tokenizer(ques, lower)
        
        # Answers   
        y1s, y2s = [], []
        answers = []
        
        for ans in record['answers']:
            ans_text = ans['text']
            ans_start = ans['answer_start']
            if ans_start != -999:
                ans_end = ans_start + len(ans_text)
                answers.append(ans_text)
                
                ans_token_idxs = []
                for token_idx, span in enumerate(spans):
                    if span[0] < ans_end and span[1] > ans_start:
                        ans_token_idxs.append(token_idx)
                y1, y2 = ans_token_idxs[0], ans_token_idxs[-1]    
                y1s.append(y1)
                y2s.append(y2)
        
        res_dict = {"context_tokens": context_tokens,
                    "ques_tokens": ques_tokens,
                    "answer": answers[0],
                    "y1s": y1s,
                    "y2s": y2s,
                    "id": count}
        count += 1
        records.append(res_dict)
    
    return records

#%%
json_dir = '/media/mynewdrive/bioqa/mnd'
with open(os.path.join(json_dir, 'MND-Intervention-1983-06Aug20.json')) as fin:
    dat = json.load(fin)

train_ls = process_records(dat['train'], lower=False)
for i, ls in enumerate(train_ls):
    train_ls[i]['id'] = 'train_' + str(train_ls[i]['id'])
    
valid_ls = process_records(dat['valid'], lower=False)
for i, ls in enumerate(valid_ls):
    valid_ls[i]['id'] = 'valid_' + str(valid_ls[i]['id'])
    
test_ls = process_records(dat['test'], lower=False)
for i, ls in enumerate(test_ls):
    test_ls[i]['id'] = 'test_' + str(test_ls[i]['id'])
    
    
    

with open(os.path.join(json_dir, 'train.json'), 'w') as fout:
    for l in train_ls:     
        fout.write(json.dumps(l) + '\n')  
        
with open(os.path.join(json_dir, 'valid.json'), 'w') as fout:
    for l in valid_ls:     
        fout.write(json.dumps(l) + '\n')   

with open(os.path.join(json_dir, 'test.json'), 'w') as fout:
    for l in test_ls:     
        fout.write(json.dumps(l) + '\n')   
