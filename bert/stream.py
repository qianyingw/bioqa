#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:20:03 2021

@author: qwang
"""
import os
os.chdir('/home/qwang/bioqa/bert')
from bert_pred import AnsPred
import streamlit as st
import io
from pyxpdf import Document
# streamlit run /home/qwang/bioqa/bert/stream.py

#%%
pth_path = '/home/qwang/bioqa/exps/psci/abs_test/best.pth.tar'


#%% App
st.header('Intervention/Induction Identification in Preclinical Text')
title = st.text_input('Input title for retrieving (if you have): ', '')
num_ans = st.number_input('Input maximum number of answer candidates: ', min_value=5, max_value=20, value=5)
upload_file = st.file_uploader("Upload your .txt or .pdf file", type=['txt', 'pdf'])
# upload_file = st.file_uploader("Upload your .txt file", type=['txt'])

if upload_file:
    if isinstance(upload_file, io.StringIO):
        context = upload_file.read()  
    if isinstance(upload_file, io.BytesIO):
        doc = Document(upload_file)
        context = doc.text()
    
    AP1 = AnsPred(context, ques='intervention') 
    if title != '':
        AP1.cut_text(title, max_sent=30)	
    AP1.pred_ans(pth_path)
    ans1 = AP1.ans_filter(num_ans)
    
    AP2 = AnsPred(context, ques='induction')
    if title != '':
        AP2.cut_text(title, max_sent=30)
    AP2.pred_ans(pth_path)
    ans2 = AP2.ans_filter(num_ans)
    
    
    st.write("""Intervention: """, {ans1[0], ans1[1], ans1[2], ans1[3], ans1[4]})
    st.write("""Induction: """, {ans2[0], ans2[1], ans2[2], ans2[3], ans2[4]})
    st.write("""Relevant text for intervention: """)	
    st.write(AP1.context)
    st.write("""Relevant text for induction: """)	
    st.write(AP2.context)
    