'''
miscellaneous functions: prob
'''
import os
import datetime
import math

import numpy as np

import torch
import torch.nn.functional as F
#from torch.autograd import Variable
#from torch.distributions import Categorical, Normal

''' stat '''
def logprob_logsumexp(logprob):
    #logprob = torch.cat(logprobs, dim=1)
    '''
    Input:
        logprob: batch_size x num_samples
    Output:
        logprob: batch_size
    '''
    # init
    num_samples = logprob.size(1)

    # eval
    logprob_maxs, _ = torch.max(logprob, dim=1, keepdim=True)
    logprob = logprob - logprob_maxs # w - \hat(w)
    logprob = torch.log(torch.sum(logprob.exp(), dim=1, keepdim=True)) # log sum(exp(w - \hat(w)))
    logprob = -math.log(float(num_samples)) + logprob_maxs + logprob # log(1/num_samples) + w + log sum(exp(w - \hat(w)))
    #logprob = torch.sum(logprob)
    return logprob.squeeze(1)
