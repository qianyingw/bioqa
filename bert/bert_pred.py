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
from sentence_transformers import SentenceTransformer, models, util

import utils
from helper.text_helper import text_cleaner, text2sents

#%%
device = torch.device('cpu') 
tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract') 
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')     
model = BertForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

#%%
class AnsPred():
    def __init__(self, context, ques):
        
        self.context = context
        if ques == 'intervention':
            self.ques = 'What is the intervention?'
        else:
            self.ques = 'What is the method of induction of disease model?'

    
    def cut_text(self, title, max_sent=30):
        '''Clean text and retrieve sents
        '''
        context = text_cleaner(self.context)  # remove intro/ref
        sents = text2sents(context, min_len=5)  # split to sent list
        ques_new = self.ques + ' ' + title
        
        # Convert sents/ques to embeds
        sent_embeds = sent_model.encode(sents, convert_to_tensor=True)
        ques_embed = sent_model.encode([ques_new], convert_to_tensor=True)
        
        # Compute cosine-similarities for each sent with other sents
        cosine_scores = util.pytorch_cos_sim(sent_embeds, ques_embed)
        # Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(cosine_scores.shape[0]):
            pairs.append({'index': i, 'score': cosine_scores[i][0]})
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
            
        sent_ls = []
        if len(sents) > max_sent:
            for pair in pairs[:max_sent]:
                sent_ls.append(sents[pair['index']])
        else:
            for pair in pairs:
                sent_ls.append(sents[pair['index']])
                
        self.context = ' '.join(sent_ls)  
        

    def pred_ans(self, pth_path):   
    
        # tokenizeation
        inputs = tokenizer(self.context, self.ques, truncation=True, max_length=512, return_tensors="pt")
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
            ans_cand.append(ans)
        
        self.ans_cand = ans_cand
    
    
    def ans_filter(self, num_ans=5):
        '''Filter answers'''
        cand = []
        for ans in self.ans_cand:
            # Remove CLS/SEP
            ans = ans.replace('[CLS]', '')
            ans = ans.replace('[SEP]', '')            
            # Strip whitespaces 
            ans = re.sub(r' - ', '-', ans)
            ans = re.sub(r'- ', '-', ans)
            ans = re.sub(r' -', '-', ans)
            ans = re.sub(r'\s+', " ", ans)
            ans = re.sub(r'^[\s]', "", ans)
            ans = re.sub(r'[\s]$', "", ans)
            # Check if candidate consists of punctuations only
            punc = [t for t in ans.split() if t in string.punctuation]  
            if len(ans) > 0 and len(ans) != len(punc):
                cand.append(ans)
                
        return cand[:num_ans]



#%%
context = """Delta-9-THC in the treatment of spasticity associated with multiple sclerosis. Marijuana is reported to decrease spasticity in patients with multiple sclerosis. This is a double blind, placebo controlled, crossover clinical trial of delta-9-THC in 13 subjects with clinical multiple sclerosis and spasticity. Subjects received escalating doses of THC in the range of 2.5-15 mg., five days of THC and five days of placebo in randomized order, divided by a two-day washout period. Subjective ratings of spasticity and side effects were completed and semiquantitative neurological examinations were performed. At doses greater than 7.5 mg there was significant improvement in patient ratings of spasticity compared to placebo. These positive findings in a treatment failure population suggest a role for THC in the treatment of spasticity in multiple sclerosis"""
title = None
pth_path = '/home/qwang/bioqa/exps/psci/abs_test/best.pth.tar'


AP = AnsPred(context, ques='intervention')
if title:
    AP.cut_text(title, max_sent=30)
AP.pred_ans(pth_path)
ans = AP.ans_filter(num_ans=5)
print(ans)
