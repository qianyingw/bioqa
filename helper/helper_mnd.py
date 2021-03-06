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
def text2tokens(text):
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


def process_for_baseline(dat):
    
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
        ans_text = record['answers'][0]['text']
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
                    "id": count}
        count += 1
        records.append(res_dict)
    
    return records

#%%
# # Intervention only
# json_dir = '/media/mynewdrive/bioqa/mnd/intervention'
# with open(os.path.join(json_dir, 'MND-Intervention-1983-06Aug20.json')) as fin:
#     dat = json.load(fin)

# # Disease only
# json_dir = '/media/mynewdrive/bioqa/mnd/disease'
# with open(os.path.join(json_dir, 'MND-Disease-1950-19Aug20.json')) as fin:
#     dat = json.load(fin)
    
# # Intervention & Disease
# json_dir = '/media/mynewdrive/bioqa/mnd/di'
# with open(os.path.join(json_dir, 'MND-DI-3933-19Aug20.json')) as fin:
#     dat = json.load(fin)


# train_ls = process_records(dat['train'], lower=False)
# for i, ls in enumerate(train_ls):
#     train_ls[i]['id'] = 'train_' + str(train_ls[i]['id'])
    
# valid_ls = process_records(dat['valid'], lower=False)
# for i, ls in enumerate(valid_ls):
#     valid_ls[i]['id'] = 'valid_' + str(valid_ls[i]['id'])
    
# test_ls = process_records(dat['test'], lower=False)
# for i, ls in enumerate(test_ls):
#     test_ls[i]['id'] = 'test_' + str(test_ls[i]['id'])
    
    
# with open(os.path.join(json_dir, 'train.json'), 'w') as fout:
#     for l in train_ls:     
#         fout.write(json.dumps(l) + '\n')  
        
# with open(os.path.join(json_dir, 'valid.json'), 'w') as fout:
#     for l in valid_ls:     
#         fout.write(json.dumps(l) + '\n')   

# with open(os.path.join(json_dir, 'test.json'), 'w') as fout:
#     for l in test_ls:     
#         fout.write(json.dumps(l) + '\n')   
