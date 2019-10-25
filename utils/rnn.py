'''
miscellaneous functions
'''
import os
import datetime

import torch


''' for bptt '''
def get_length_bptt_batch_generator_sequence_modeling(source, bptt):
    return len(range(0, source.size(0) - 1, bptt))

def bptt_batch_generator_sequence_modeling(source, bptt):
    '''
    Copied and modified from https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L111

    source: sequence length x batch size

    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    '''
    for i in range(0, source.size(0) - 1, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len]#.view(-1)
        yield data, target

def get_length_bptt_batch_generator_latent_variable_models(source, bptt):
    return len(range(0, source.size(0), bptt))

def bptt_batch_generator_latent_variable_models(source, bptt):
    '''
    Copied and modified from bptt_batch_generator_sequence_modeling

    source: sequence length x batch size

    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ a g m s ┐
    └ b h n t ┘ └ b h n t ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    '''
    for i in range(0, source.size(0), bptt):
        seq_len = min(bptt, len(source) - i)
        data = source[i:i+seq_len]
        target = source[i:i+seq_len]#.view(-1)
        yield data, target


''' for rnn interface '''
def pack_hiddens(hiddens, rnn_type):
    '''
    Input:
    hiddens: a list with length = number of layers (of stacked RNNs)
             each element in the list: (h_t, c_t) (or h_t) of each RNN

    Output:
    new_hiddens: (h_t, c_t) = a tuple if lstm
                 h_t = a tensor else
                 where h_t (and c_t) is tensor num_layers x batch_size x hidden_size
    '''
    if rnn_type == 'LSTM':
        h_t = []
        c_t = []
        for hidden in hiddens:
            h_t.append(hidden[0].unsqueeze(0))
            c_t.append(hidden[1].unsqueeze(0))
        h_t = torch.cat(h_t, dim=0)
        c_t = torch.cat(c_t, dim=0)
        return (h_t, c_t)
    else:
        h_t = []
        for hidden in hiddens:
            h_t.append(hidden[0].unsqueeze(0))
        h_t = torch.cat(h_t, dim=0)
        return h_t

def unpack_hiddens(hiddens, rnn_type):
    '''
    Input:
    hiddens: (h_t, c_t) = a tuple if lstm
             h_t = a tensor else
             where h_t (and c_t) is tensor num_layers x batch_size x hidden_size

    Output:
    new_hiddens: a list with length = number of layers (of stacked RNNs)
                 each element in the list: (h_t, c_t) (or h_t) of each RNN
    '''
    num_layers = hiddens[0].size(0) if rnn_type == 'LSTM' else hiddens.size(0)
    new_hiddens = []
    for j in range(num_layers):
        if rnn_type == 'LSTM':
            h_t = hiddens[0].narrow(0, j, 1).squeeze(0)
            c_t = hiddens[1].narrow(0, j, 1).squeeze(0)
            hidden = (h_t, c_t)
        else:
            h_t = hiddens.narrow(0, j, 1).squeeze(0)
            hidden = h_t
        new_hiddens.append(hidden)
    return new_hiddens
