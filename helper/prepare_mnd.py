#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:17:30 2020

@author: qwang
"""

from sklearn.utils import shuffle
import pandas as pd
import re
import random
import json

SEED = 1234
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
random.seed(SEED)



#%% Read MND Disease/Intervention root-only data
mnd = pd.read_csv('/media/mynewdrive/bioqa/mnd/MND_DI_RootOnly_28Jul2020.csv', sep=',', engine="python")  # 8586

# Read all title and abstracts with utf-8 encoding
tiabs = pd.read_csv('/media/mynewdrive/bioqa/mnd/myStudies_TiAbs_05Aug2020.csv', sep=',', engine="python")  # 20770
tiabs = tiabs.rename(columns={"idStr": "StudyIdStr"})  # 'idStr' in myStudies.csv matches 'studyIdStr' in fullAnnotations.csv

# Replace by cleaner title and abstract
mnd = pd.merge(mnd, tiabs, how='inner', on=['StudyIdStr'])  # 8566
mnd = mnd.rename(columns={"Title_y": "Title", "Abstract_y": "Abstract"})
list(mnd.columns)

# Remove records without abstract and investigator info
mnd = mnd.dropna(subset=['Abstract', 'Investigator'], how='any')  # 8490
# mnd = mnd[mnd.QuestionIdStr.isin(['f0b87908-4fe0-48d2-bf09-eda9d677ed75', 'e575bef7-a84b-4f3e-a5be-21107f797e0c'])]  # 8490


#%%
def remove_duplicates(df_sub):
    ''' Extract non-duplicate records with only one annotation or duplicate records with exact same annotations
        df_ndup: records with unique StudyIdStr
    '''
    study_gp = list(df_sub.groupby(['StudyIdStr']))
    ndup = []
    for i in range(len(study_gp)):
        df = study_gp[i][1]
        if len(df) == 1 or (len(df) > 1 and len(set(df['Answer'])) == 1):
            ndup.append(study_gp[i])
    
    frames = [dg[1] for dg in ndup]
    df_ndup = pd.concat(frames) 
    # Drop records with duplicate annotations from multiple investigators
    df_ndup = df_ndup.drop_duplicates(subset=['StudyIdStr'])
    
    # Combine title and abstract
    df_ndup['Context'] = df_ndup['Title'] + '. ' + df_ndup["Abstract"]
    df_ndup = df_ndup[['StudyIdStr', 'QuestionIdStr', 'Question', 'Answer', 'Context']]
    df_ndup = df_ndup.reset_index(drop=True)
    
    return df_ndup


def get_ans_char_pos(df_ndup):
    ''' Check number of answer matches and save char positions of first 3 matches
        df_ndup: records with unique StudyIdStr
    '''
    for i, row in df_ndup.iterrows():  
        regex = r'\b' + re.escape(row['Answer']) + r'\b'
        matches = re.finditer(regex, row['Context'], re.MULTILINE)
        n_matches = 0 
        for matchId, match in enumerate(matches, start=1):
            n_matches += 1
            if n_matches == 1:
                df_ndup.loc[i,'ans_start1'] = match.start()
            if n_matches == 2:
                df_ndup.loc[i,'ans_start2'] = match.start()
            if n_matches == 3:
                df_ndup.loc[i,'ans_start3'] = match.start()
        df_ndup.loc[i,'numMatches'] = n_matches
            
    # Remove records with '0' matches
    df_ndup = df_ndup[df_ndup['numMatches'] != 0]  # 1983       
    # Fill nan with -999
    df_ndup['ans_start1'] = df_ndup['ans_start1'].fillna(-999)
    df_ndup['ans_start2'] = df_ndup['ans_start2'].fillna(-999)
    df_ndup['ans_start3'] = df_ndup['ans_start3'].fillna(-999)
    
    return df_ndup

#%% Process intervention records
inter = mnd[mnd.QuestionIdStr=='f0b87908-4fe0-48d2-bf09-eda9d677ed75']  # 4248
# Strip whitespaces
inter['Answer'] = inter['Answer'].str.strip()
len(set(inter['StudyIdStr']))  # 3100
len(set(inter['Answer']))  # 2746

# Obtain records with only one annotation or de-duplicate records with same annotations from multiple investigators
inter_ndup = remove_duplicates(inter)  # 2476
# Check number of answer matches and save char positions of first 3 matches
inter_ans = get_ans_char_pos(inter_ndup)  # 1983


#%% Process disease records
disea = mnd[mnd.QuestionIdStr=='e575bef7-a84b-4f3e-a5be-21107f797e0c']  # 4242
# Strip whitespaces
disea['Answer'] = disea['Answer'].str.strip()
len(set(disea['StudyIdStr']))  # 3099
len(set(disea['Answer']))  # 6

# Obtain records with only one annotation or de-duplicate records with same annotations from multiple investigators
disea_ndup = remove_duplicates(disea)  # 3092
# Check number of answer matches and save char positions of first 3 matches
disea_ans = get_ans_char_pos(disea_ndup)  # 1950


#%% Concate intervention & disease dataframe
dat = pd.concat([inter_ans, disea_ans], sort=False)  # 3933 = 1983 + 1950                                                 
list(dat.columns)                                             

#%% Split data to train/valid/test
dat = shuffle(dat)
dat = dat.reset_index(drop=True)

dat['Group'] = 'test'
train_size = int(TRAIN_RATIO*len(dat))
val_size = int(VALID_RATIO*len(dat))
dat.loc[:train_size, 'Group'] = 'train'
dat.loc[train_size : (train_size+val_size), 'Group'] = 'valid'
    
# dat.to_csv('dat.csv', sep=',', encoding='utf-8')

#%% Create output json file
def record_list(df, group):
    sub = df[df['Group'] == group]
    sub = sub.reset_index(drop=True)
    ls = []
    for i, row in sub.iterrows():
        record = {
			"question": row['Question'],
			"QuestionIdStr": row['QuestionIdStr'],
			"StudyIdStr": row['StudyIdStr'],
			"answers": [
				{"text": row['Answer'], "answer_start": int(row['ans_start1'])},
				{"text": row['Answer'], "answer_start": int(row['ans_start2'])},
				{"text": row['Answer'], "answer_start": int(row['ans_start3'])}
			],
			"context": row['Context']}
        ls.append(record)
    return ls

train_ls = record_list(dat, group='train')
valid_ls = record_list(dat, group='valid')
test_ls = record_list(dat, group='test')

out = {"train": train_ls,
       "valid": valid_ls,
       "test": test_ls}

with open('/media/mynewdrive/bioqa/mnd/di/MND-DI-3933-19Aug20.json', 'w') as fout: 
    fout.write(json.dumps(out))
    
# with open('/media/mynewdrive/bioqa/mnd/intervention/MND-Intervention-1983-06Aug20.json', 'w') as fout: 
#     fout.write(json.dumps(out))
        
# with open('/media/mynewdrive/bioqa/mnd/disease/MND-Disease-1950-19Aug20.json', 'w') as fout: 
#     fout.write(json.dumps(out))        
        
        
        
#%% (Not in use) Original processing for intervention data
# Intervention records
inter = mnd[mnd.QuestionIdStr=='f0b87908-4fe0-48d2-bf09-eda9d677ed75']  # 4248

# Strip whitespaces
inter['Answer'] = inter['Answer'].str.strip()
len(set(inter['StudyIdStr']))  # 3100
len(set(inter['Answer']))  # 2746

study_gp = inter.groupby(['StudyIdStr'])
study_gp = list(study_gp)  # 3100

ndup, dup2, dup3, dup4, dup5 = [], [], [], [], []
for i in range(len(study_gp)):
    df = study_gp[i][1]
    if len(df) == 2 and len(set(df['Answer'])) != 1:
        dup2.append(study_gp[i])
    elif len(df) == 3 and len(set(df['Answer'])) != 1:
        dup3.append(study_gp[i])
    elif len(df) == 4 and len(set(df['Answer'])) != 1:
        dup4.append(study_gp[i])
    elif len(df) == 5 and len(set(df['Answer'])) != 1:
        dup5.append(study_gp[i])
    else:  
        ndup.append(study_gp[i])
        
# len(ndup) = 2476
# 2476+405+185+33+1 = 3100

# frames = [dg[1] for dg in dup2]
# dup_df = pd.concat(frames)
# dup_df.to_csv('dup2.csv', sep=',', encoding='utf-8')
del(dup2); del(dup3); del(dup4); del(dup5)

# Select records with annotation from only one investigator or exact same annotations from multiple investigators
frames = [dg[1] for dg in ndup]
df = pd.concat(frames)  # 2746
# Drop duplicate annotations from multiple investigators
df_ndup = df.drop_duplicates(subset=['StudyIdStr'])  # 2476

# Combine title and abstract
df_ndup['Context'] = df_ndup['Title'] + '. ' + df_ndup["Abstract"]
inv = df_ndup[['StudyIdStr', 'QuestionIdStr', 'Question', 'Answer', 'Context']]
inv = inv.reset_index(drop=True)

# Check number of matches and save positions of first 3 matches (answer in the context)
for i, row in inv.iterrows():  
    regex = r'\b' + re.escape(row['Answer']) + r'\b'
    matches = re.finditer(regex, row['Context'], re.MULTILINE)
    n_matches = 0 
    for matchId, match in enumerate(matches, start=1):
        n_matches += 1
        if n_matches == 1:
            inv.loc[i,'ans_start1'] = match.start()
        if n_matches == 2:
            inv.loc[i,'ans_start2'] = match.start()
        if n_matches == 3:
            inv.loc[i,'ans_start3'] = match.start()
    inv.loc[i,'numMatches'] = n_matches
        
# Remove records with '0' matches
inv = inv[inv['numMatches'] != 0]  # 1983       
       
# Fill nan with -999
inv['ans_start1'] = inv['ans_start1'].fillna(-999)
inv['ans_start2'] = inv['ans_start2'].fillna(-999)
inv['ans_start3'] = inv['ans_start3'].fillna(-999)