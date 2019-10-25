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
def logprob_gaussian(mu, logvar, z, do_sum=True):
    '''
    Inputs: 
        z: b x nz
        mu, logvar: b x nz
    Outputs:
        prob: b x nz
    '''
    # logprob
    neglogprob = (z - mu)**2 / logvar.exp() + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    #if do_mean:
    #    logprob = torch.mean(logprob, dim=1, keepdim=True)
    if do_sum:
        assert NotImplementedError
    else:
        batch_size = logprob.size(0)
        logprob = torch.sum(logprob.view(batch_size, -1), dim=1)
    return logprob

def logprob_gaussian_w_fixed_var(mu, z, std=1.0, do_sum=True):
    '''
    Inputs: 
        z: b x nz
        mu, logvar: b x nz
    Outputs:
        prob: b x nz
    '''
    # init var, logvar
    var = std**2
    logvar = math.log(var)

    # logprob
    neglogprob = (z - mu)**2 / var + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    #if do_mean:
    #    logprob = torch.mean(logprob, dim=1, keepdim=True)
    if do_sum:
        assert NotImplementedError
    else:
        batch_size = logprob.size(0)
        logprob = torch.sum(logprob.view(batch_size, -1), dim=1)
    return logprob

def logprob_logsumexp(logprob):
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
     return logprob.squeeze(1)
