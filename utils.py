#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:23:05 2020

@author: qwang
"""

import torch
import torch.nn.functional as F


#%%
def masked_softmax(p, mask, dim=-1, log_softmax=False):
    """
    Take the softmax of `p` over given dimension, and set entries to 0 wherever `mask` is 0.
    Args:
        p (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `p`, with 0 indicating positions that should be assigned 0 probability in the output.
        log_softmax: Take log-softmax rather than regular softmax because `F.nll_loss` expect log-softmax.

    Returns:
        p (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    # If mask = 0, masked_p = 0 - 1e30 (~=-inf)
    # If mask = 1, masked_p = p
    masked_p = mask * p + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    p = softmax_fn(masked_p, dim)

    return p