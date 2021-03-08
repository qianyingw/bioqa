#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:37:30 2021

@author: qwang
"""

import os
os.chdir('/home/qwang/bioqa')

import re
import string
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

import utils

#%%
device = torch.device('cpu') 
tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')    
model = BertForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

#%%
def pred_ans(context, question, pth_path, num_ans=5):   
    
    # tokenizeation
    inputs = tokenizer(context, question, truncation=True, max_length=512, return_tensors="pt")
    # inputs = tokenizer(context, question, truncation=True, max_length=512, do_lower_case=False, return_tensors="pt")
    # _batch_encode_plus() got an unexpected keyword argument 'do_lower_case'
     
    # Load checkpoin
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cpu()
    
    ## Run model
    model.eval()
    outputs = model(**inputs) 
    p1s, p2s = outputs[0], outputs[1]  # [1, clen+qlen]
    
    ## Get start/end idxs
    p1s = utils.masked_softmax(p1s, 1-inputs['token_type_ids'], dim=1, log_softmax=True)  # [1, clen+qlen]
    p2s = utils.masked_softmax(p2s, 1-inputs['token_type_ids'], dim=1, log_softmax=True)  # [1, clen+qlen]
    p1s, p2s = p1s.exp(), p2s.exp()
    s_idxs, e_idxs, top_probs = utils.get_ans_list_idx(p1s, p2s, num_answer=30)  # [1, num_answer]
    
    ### Get ans candidates ###
    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])   
    ans_cand = []
    for j in range(30):  # iter candidates                      
        ans_jth_tokens = all_tokens[s_idxs[0, j]: (e_idxs[0, j]+1)]  # Token list of one answer candidate
        ans_ids = tokenizer.convert_tokens_to_ids(ans_jth_tokens)
        ans = tokenizer.decode(ans_ids)
               
        ### Answer filter ###
        ans = ans.replace('[CLS]', '')
        ans = ans.replace('[SEP]', '')
        # Strip whitespaces 
        ans = re.sub(r'\s+', " ", ans)
        ans = re.sub(r'^[\s]', "", ans)
        ans = re.sub(r'[\s]$', "", ans)
        # Check if candidate consists of punctuations only
        punc = [t for t in ans.split() if t in string.punctuation]  
        if len(ans) > 0 and len(ans) != len(punc):
            ans_cand.append(ans)
            
    return ans_cand[:num_ans]
    
#%%
context = """Delta-9-THC in the treatment of spasticity associated with multiple sclerosis. Marijuana is reported to decrease spasticity in patients with multiple sclerosis. This is a double blind, placebo controlled, crossover clinical trial of delta-9-THC in 13 subjects with clinical multiple sclerosis and spasticity. Subjects received escalating doses of THC in the range of 2.5-15 mg., five days of THC and five days of placebo in randomized order, divided by a two-day washout period. Subjective ratings of spasticity and side effects were completed and semiquantitative neurological examinations were performed. At doses greater than 7.5 mg there was significant improvement in patient ratings of spasticity compared to placebo. These positive findings in a treatment failure population suggest a role for THC in the treatment of spasticity in multiple sclerosis"""
question ="What is the intervention?"
pth_path = '/home/qwang/bioqa/exps/abs_test/best.pth.tar'

print(pred_ans(context, question, pth_path, num_ans=5))
