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

#%%
# import utils
# from helper.text_helper import text_cleaner, text2sents
import numpy as np
import torch.nn.functional as F

def masked_softmax(p, mask, dim=-1, log_softmax=False):
    """ Take the softmax of `p` over given dimension, and set entries to 0 wherever `mask` is 0.
    """
    mask = mask.type(torch.float32)
    # If mask = 0, masked_p = 0 - 1e30 (~=-inf)
    # If mask = 1, masked_p = p
    masked_p = mask * p + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    p = softmax_fn(masked_p, dim)

    return p
	

def get_ans_list_idx(p_s, p_e, max_len=15, threshold=None, num_answer=1):
    """
    Discretize soft predictions (probs) to get start and end indices
    Choose top [num_answer] pairs of (i, j) where p1[i]*p2[j] > threshold, s.t. (i <= j) & (j-i+1 <= max_len)
    """
    c_len = p_s.shape[1]
    device = p_s.device
    
    if p_s.min() < 0 or p_s.max() > 1 or p_e.min() < 0 or p_e.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probs
    p_s = p_s.unsqueeze(2)  # [batch_size, c_len, 1]
    p_e = p_e.unsqueeze(1)  # [batch_size, 1, c_len]    
    p_join = torch.bmm(p_s, p_e)  # [batch_size, c_len, c_len]
    
    # Restrict (i, j) s.t. (i <= j) & (j-i+1 <= max_len)
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair = is_legal_pair - torch.triu(torch.ones((c_len, c_len), device=device), diagonal=max_len)
    p_join = p_join * is_legal_pair
        
    if threshold:  
        batch_s_idxs, batch_e_idxs = [], []
        for p in p_join:            
            is_larger = p > threshold  # p: [c_len, c_len]
            p = p * is_larger
        
            if num_answer > 1 and torch.sum(is_larger) >= num_answer:
                # Obtain top [num_answer] (i, j) pairs where p1[i]*p2[j] > threshold
                _, flat_idxs = torch.topk(torch.flatten(p, start_dim=0), num_answer)  # [num_answer]
                idxs = torch.tensor(np.unravel_index(flat_idxs.numpy(), p.shape))  # [2, num_answer]
                s_idxs, e_idxs = idxs[0], idxs[1]            
            else:
                # Obtain (i, j) pairs where p1[i]*p2[j] > threshold
                # len(s_idxs) = 0 when torch.sum(is_larger) = 0
                s_idxs, e_idxs = torch.where(is_larger==True)  # List of answer start/end idxs for one record
                
            batch_s_idxs.append(s_idxs.tolist()) 
            batch_e_idxs.append(e_idxs.tolist())  
            
        return batch_s_idxs, batch_e_idxs  # list: element is list of ans idxs
       
    else:
        # Obtain top [num_answer] (i, j) pairs which maximize p_join
        top_probs, flat_idxs = torch.topk(torch.flatten(p_join, start_dim=1), num_answer)  # [batch_size, num_answer]
        
        s_idxs, e_idxs = [], []
        for fidx in flat_idxs:
            idxs = torch.tensor(np.unravel_index(fidx.numpy(), p_join.shape[1:]))  # [2, num_answer]
            s_idxs.append(idxs[0])  # idxs[0]: [num_answer]
            e_idxs.append(idxs[1])  # idxs[1]: [num_answer]
            
        s_idxs = torch.stack(s_idxs)    
        e_idxs = torch.stack(e_idxs)            
        return s_idxs, e_idxs, torch.sqrt(top_probs)  # tensor, [batch_size, num_answer]

import re
import spacy
nlp = spacy.load("en_core_sci_sm")
# Text cleaning
p_ref = re.compile(r"(.*Reference\s{0,}\n)|(.*References\s{0,}\n)|(.*Reference list\s{0,}\n)|(.*REFERENCE\s{0,}\n)|(.*REFERENCES\s{0,}\n)|(.*REFERENCE LIST\s{0,}\n)", 
                   flags=re.DOTALL)
def text_cleaner(text):
    # Remove texts before the first occurence of 'Introduction' or 'INTRODUCTION'
    text = re.sub(r".*?(Introduction|INTRODUCTION)\s{0,}\n{1,}", " ", text, count=1, flags=re.DOTALL)    
    # Remove reference after the last occurence 
    s = re.search(p_ref, text)
    if s: text = s[0]  
    # Remove emtpy lines
    text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)
    # Remove lines with digits/(digits,punctuations,line character) only
    text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", text)
    # Remove non-ascii characters
    text = text.encode("ascii", errors="ignore").decode()     
    # Strip whitespaces 
    text = re.sub(r'\s+', " ", text)
    # Remove the whitespace at start and end of line
    text = re.sub(r'^[\s]', "", text)
    text = re.sub(r'[\s]$', "", text)
    return text

# Split text to sentences
def text2sents(text, min_len):
    doc = nlp(text)
    # Convert spacy span to string list 
    sents = list(doc.sents)  
    sents = [str(s) for s in sents]     
    # Remove too short sentences
    sents = [s for s in sents if len(s.split(' ')) > min_len]  # 296
    return sents


#%%
device = torch.device('cpu') 
tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract') 
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')     
model = BertForQuestionAnswering.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

#%%
from spacy.lang.en.stop_words import STOP_WORDS
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
        p1s = masked_softmax(p1s, 1-inputs['token_type_ids'], dim=1, log_softmax=True)  # [1, clen+qlen]
        p2s = masked_softmax(p2s, 1-inputs['token_type_ids'], dim=1, log_softmax=True)  # [1, clen+qlen]
        p1s, p2s = p1s.exp(), p2s.exp()
        s_idxs, e_idxs, top_probs = get_ans_list_idx(p1s, p2s, max_len=5, num_answer=30)  # [1, num_answer]
        
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
            # Check if candidate consists of punctuations/stopwords/digits only
            punc = [t for t in ans.split() if t in string.punctuation+string.digits]  
            stop = [t for t in ans.split() if t in STOP_WORDS]
            if len(ans) > 0 and len(ans.split()) != len(punc) and len(ans.split()) != len(stop):
                if ans not in cand: 
                    cand.append(ans)
                 
        return cand[:num_ans]



#%% Extract from abstract
# # PMC3288606
# context = '''The dose-limiting side-effect of taxane, platinum-complex, and other kinds of anti-cancer drugs is a chronic, distal, bilaterally symmetrical, sensory peripheral neuropathy that is often accompanied by neuropathic pain. Work with animal models of these conditions suggests that the neuropathy is a consequence of toxic effects on mitochondria in primary afferent sensory neurons. If this is true, then additional mitochondrial insult ought to make the neuropathic pain worse. This prediction was tested in rats with painful peripheral neuropathy due to the taxane agent, paclitaxel, and the platinum-complex agent, oxaliplatin. Rats with established neuropathy were given one of three mitochondrial poisons: rotenone (an inhibitor of respiratory Complex I), oligomycin (an inhibitor of ATP synthase), and auranofin (an inhibitor of the thioredoxin-thioredoxin reductase mitochondrial anti-oxidant defense system). All three toxins significantly increased the severity of paclitaxel-evoked and oxaliplatin-evoked mechano-allodynia and mechano-hyperalgesia while having no effect on the mechano-sensitivity of chemotherapy na?ve rats. Chemotherapy-evoked painful peripheral neuropathy is associated with an abnormal spontaneous discharge in primary afferent A-fibers and C-fibers. Oligomycin, at the same dose that exacerbated allodynia and hyperalgesia, significantly increased the discharge frequency of spontaneously discharging A-fibers and C-fibers in both paclitaxel-treated and oxaliplatin-treated rats, but did not evoke any discharge in na?ve control rats. These results implicate mitochondrial dysfunction in the production of chemotherapy-evoked neuropathic pain and suggest that drugs that have positive effects on mitochondrial function may be of use in its treatment and prevention.'''
# title = None
# pth_path = '/home/qwang/bioqa/exps/psci/abs_test/best.pth.tar'

# AP = AnsPred(context, ques='intervention')
# AP.pred_ans(pth_path)
# ans = AP.ans_filter(num_ans=5)
# print(ans)

# #%% Extract from full text
# # pdftotext /home/qwang/bioqa/bert/PMC3288606.pdf /home/qwang/bioqa/bert/PMC3288606.txt
# with open("bert/PMC3288606.txt", 'r', encoding='utf-8', errors='ignore') as fin:
#     context = fin.read()  
# title = "Effects of mitochondrial poisons on the neuropathic pain produced by the chemotherapeutic agents, paclitaxel and oxaliplatin."
# pth_path = '/home/qwang/bioqa/exps/psci/abs_test/best.pth.tar'


# AP = AnsPred(context, ques='intervention')
# if title:
#     AP.cut_text(title, max_sent=30)
# AP.pred_ans(pth_path)
# ans = AP.ans_filter(num_ans=10)
# print(ans)
