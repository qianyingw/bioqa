#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:52:47 2020

@author: qwang
"""


import pandas as pd
import os
import re

#%% Read CIPN data from Gill
cipn = pd.read_csv('/media/mynewdrive/bioqa/cipn/Dataset2_InterventionAll_Nested.csv', sep=',', engine="python")  # 1397

list(cipn.columns)
len(set(cipn['Pub ID']))  # 216
cipn = cipn[['Pub ID', 'Group letter', 'Units', 
             'Animal', 'Strain', 'Sex',
             'Method of Induction of neuropathic pain', 'Dose/severity to induce model',
             'Drug', 'Dose']]

#%% Read CIPN rob data from Jing (with path link)
link = pd.read_csv('/media/mynewdrive/bioqa/cipn/np data rob output - double screened.csv', 
                   usecols=['Pub ID', 'DocumentLink'], sep=',', engine="python")  # 4453
link = link.dropna()  # 4323
link['DocumentLink'] = link['DocumentLink'].str.replace(r"(#\\.*)|(\\.*)|(#)", "")


link = link.drop_duplicates()  # 550
len(set(link['Pub ID']))  # 547
link = link.drop_duplicates()  # 550

# Check duplicate "Pub ID" and "DocumentLink"
dup = []
gp = list(link.groupby(['Pub ID']))
for i in range(len(gp)):
    df = gp[i][1]
    if len(df) > 1:
        dup.append(gp[i])

frames = [dg[1] for dg in dup]
df_dup = pd.concat(frames) 


# Remove invalid DocumentLink
link = link[-link["Pub ID"].isin([20668, 404641, 420986])]
link = link.append({"Pub ID": 404641, "DocumentLink": "Publications/NP_references/1507778383/Fariello-2014-Broad spectrum and prolonged eff.pdf"}, ignore_index=True)
link = link.append({"Pub ID": 420986, "DocumentLink": "Publications/NP_references/420986_Zhao_2014.pdf"}, ignore_index=True)  # 546


#%% Merge DocumentLink to CIPN data 
cipn = pd.merge(cipn, link, how='left', on=['Pub ID'])  # 1397
cipn = cipn.rename(columns={"Pub ID": "PubID", 
                            "Group letter": "GroupLetter",
                            "Method of Induction of neuropathic pain": "InductionMethod",
                            "Dose/severity to induce model": "InductionDose"})

cipn = cipn.dropna(subset=['DocumentLink'])  # 1386
cipn = cipn.reset_index(drop=True)
len(set(cipn['PubID']))  # 215

# cipn.to_csv('/media/mynewdrive/bioqa/cipn/Dataset2_InterventionAll_Nested_DocLink.csv', sep=',', encoding='utf-8')

# Group records by paper IDs
pub_gp = list(cipn.groupby(['PubID']))  # 215
TXT_DIR = "/media/mynewdrive/bioqa/cipn/TiAbs"
# cipn['PaperGroupID'] = cipn['PubID'] + '_' + cipn['GroupLetter'] 

#%% Generate record list

def cipn_list(var_name):
    '''
    Parameters
        var_name : column name
    Returns
        record_ls : dictionary list
        
    For each paepr
        Read title/abstract 
        Obtain list of answers and chart positions of first match
        Remove answers without matching
        Remove paper when all answers can't be matched
    '''
    record_ls = []
    
    for i in range(len(pub_gp)):
        # For each individual paper
        df = pub_gp[i][1]
        
        # Read title/abstract
        txtpath = os.path.join(TXT_DIR, str(df['PubID'].values[0])+'.txt')    
        if os.path.isfile(txtpath): 
            with open(txtpath, 'r', encoding='utf-8', errors='ignore') as fin:
                context = fin.read()
            ans_starts = []
            answers = list(set(df[var_name]))
            
            # Take first exact match 
            final_answers = []
            for ans in answers:
                regex = r'\b' + re.escape(ans) + r'\b'
                matches = re.finditer(regex, context, re.MULTILINE)
                n_matches = 0
                for matchId, match in enumerate(matches, start=1):
                    n_matches += 1
                    if n_matches == 1:
                        # Add ans to final answer list only when there is a match
                        final_answers.append(ans)  
                        ans_starts.append(match.start() - 1)  # because of utf-8 encoding, context[0] is '\ufeff' rather than the 1st char
                        break    
                # No matches
                # if n_matches == 0:
                #     ans_starts.append(-999)
            
            # Skip paper if there is no matching for all answers
            if len(final_answers) > 0:      
                record = {
                    "question": "What is the intervention?",
             		"pid": df['PubID'].values[0],
             		"answers": {
                         "text": answers, 
                         "answer_start": ans_starts},
             		"context": context}
            
            record_ls.append(record)
        
    return record_ls


# Drug
drug_ls = cipn_list('Drug')  # 213
# Method of Model Induction
induce_ls = cipn_list('InductionMethod')  # 213





