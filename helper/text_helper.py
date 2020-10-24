#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:23:27 2020

@author: qwang
"""
import re
import spacy

nlp = spacy.load("en_core_sci_sm")

#%% Text cleaning
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

#%% Split text to sentences
def text2sents(text, min_len):
    doc = nlp(text)
    # Convert spacy span to string list 
    sents = list(doc.sents)  
    sents = [str(s) for s in sents]     
    # Remove too short sentences
    sents = [s for s in sents if len(s.split(' ')) > min_len]  # 296
    return sents


#%% Split text to word tokens
def text2tokens(text):
    tokens = [token.text for token in nlp(text)]
    return tokens
    
    