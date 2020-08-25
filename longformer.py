#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:47:12 2020

@author: qwang
"""

import torch
from transformers import LongformerTokenizerFast


#%%
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

question, text = "Who is Kim Seokjin?", "Kim Seokjin is a famous singer in South Korea."
encoding = tokenizer(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attn_mask = encoding["attention_mask"]
outputs = model(input_ids, attention_mask = attn_mask)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token