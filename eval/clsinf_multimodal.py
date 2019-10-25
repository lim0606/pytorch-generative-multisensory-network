# import
import os
import argparse
import time
import math
import itertools

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import datasets as dset
import models as net

from utils import get_time, logging, get_lrs
from utils import get_grid_image, get_image_from_values, get_plot
from utils import batch_to_device, merge_two_batch, get_masks#, trim_context_target
from utils.gqn_tmp import new_trim_context_target as trim_context_target
from utils import sample_queries, sample_hand_queries, sample_random_queries, sample_random_hand_queries
from utils import get_visualization_image_data, get_visualization_haptic_data, get_combined_visualization_image_data
from utils import logprob_logsumexp


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='haptix-shepard_metzler_5_parts', help='dataset')

# net architecture
parser.add_argument('--model', default='gqn',
                    choices=['multimodal-gqn-v1', 'multimodal-gqn-v2', 'multimodal-gqn-v3', 'multimodal-gqn-v4', 'multimodal-ml-gqn-v1', 'multimodal-ml-gqn-v2',
                             'multimodal-cgqn-v1', 'multimodal-cgqn-v2', 'multimodal-cgqn-v3', 'multimodal-cgqn-v4', 'multimodal-ml-cgqn-v1', 'multimodal-ml-cgqn-v2',
                             'poe-multimodal-cgqn-v1', 'poe-multimodal-cgqn-v2', 'poe-multimodal-cgqn-v3', 'poe-multimodal-cgqn-v4', 'poe-multimodal-ml-cgqn-v1', 'poe-multimodal-ml-cgqn-v2',
                             'apoe-multimodal-cgqn-v1', 'apoe-multimodal-cgqn-v2', 'apoe-multimodal-cgqn-v3', 'apoe-multimodal-cgqn-v4', 'apoe-multimodal-ml-cgqn-v1', 'apoe-multimodal-ml-cgqn-v2',
                             'conv-apoe-multimodal-cgqn-v1', 'conv-apoe-multimodal-cgqn-v2', 'conv-apoe-multimodal-cgqn-v3', 'conv-apoe-multimodal-cgqn-v4', 'conv-apoe-multimodal-ml-cgqn-v1', 'conv-apoe-multimodal-ml-cgqn-v2',
                             'cond-conv-apoe-multimodal-cgqn-v1', 'cond-conv-apoe-multimodal-cgqn-v2', 'cond-conv-apoe-multimodal-cgqn-v3', 'cond-conv-apoe-multimodal-cgqn-v4', 'cond-conv-apoe-multimodal-ml-cgqn-v1', 'cond-conv-apoe-multimodal-ml-cgqn-v2',
                             ],
                    help='model')

# type of data
parser.add_argument('--img-nheight', type=int, default=64,
                    help='the height / width of the input to network')
parser.add_argument('--img-nchannels', type=int, default=3,
                    help='number of channels in input')
parser.add_argument('--hpt-nheight', type=int, default=1,
                    help='the height / width of the input to network')
parser.add_argument('--hpt-nchannels', type=int, default=132,
                    help='number of channels in input')

# eval
parser.add_argument('--mod-step', type=int, default=3,
                    help='the height / width of the input to network')
parser.add_argument('--mask-step', type=int, default=5,
                    help='the height / width of the input to network')

# training
parser.add_argument('--train-batch-size', type=int, default=10,
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=10,
                    help='input batch size for test (default: 10)')
parser.add_argument('--num-z-samples', type=int, default=50,
                    help='number of latent samples to estimate logprob')
parser.add_argument('--num-q-samples', type=int, default=1,
                    help='number of latent samples to estimate logprob')
parser.add_argument('--num-iters', type=int, default=2000,
                    help='number of iters used for evaluation; thus, the number of evaluated data = num_iters * eval_batch_size')

# eval mode
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='eval image data in grayscale')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cache', default='eval2', help='path to cache')
parser.add_argument('--path', type=str, default='./model.pt',
                    help='model file to load')
parser.add_argument('--vis-interval', type=int, default=10,
                    help='visualization interval')

# parse arguments
opt = parser.parse_args()
opt.new_path = os.path.join(opt.cache, opt.path)
os.system('mkdir -p {}'.format(opt.new_path))

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if opt.cuda else "cpu")

# print args
logging(str(opt), path=opt.new_path)

# init tensorboard
writer = SummaryWriter(opt.new_path)


# init dataset
train_loader, val1_loader, val2_loader, test_loader, dataset_info = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda)
num_modalities = dataset_info['num_modalities']
mask_combinations = train_loader.dataset.combinations
num_combinations = len(mask_combinations)
num_data_types, indices_data_types, data_types = 0, {}, {}
for channels, height, width, nc_query, mtype in dataset_info['dims']:
    if not mtype in data_types:
        data_types[mtype] = 1
        indices_data_types[mtype] = num_data_types
        num_data_types += 1
    else:
        data_types[mtype] += 1
dims = []
for i, (channels, height, width, nc_query, mtype) in enumerate(dataset_info['dims']):
    dim = channels*height*width
    dims += [dim]
nviews = dataset_info['nviews']


# init model
if opt.model == 'multimodal-cgqn-v1':
    model = net.MultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'multimodal-cgqn-v2':
    model = net.MultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'multimodal-cgqn-v3':
    model = net.MultimodalCGQN_v3(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'multimodal-cgqn-v4':
    model = net.MultimodalCGQN_v4(dims=dataset_info['dims'], num_layers=1).to(device)

elif opt.model == 'multimodal-ml-cgqn-v1':
    model = net.MultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=2).to(device)
elif opt.model == 'multimodal-ml-cgqn-v2':
    model = net.MultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=2).to(device)

elif opt.model == 'poe-multimodal-cgqn-v1':
    model = net.PoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'poe-multimodal-cgqn-v2':
    model = net.PoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'poe-multimodal-cgqn-v3':
    model = net.PoEMultimodalCGQN_v3(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'poe-multimodal-cgqn-v4':
    model = net.PoEMultimodalCGQN_v4(dims=dataset_info['dims'], num_layers=1).to(device)

elif opt.model == 'poe-multimodal-ml-cgqn-v1':
    model = net.PoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=2).to(device)
elif opt.model == 'poe-multimodal-ml-cgqn-v2':
    model = net.PoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=2).to(device)

elif opt.model == 'apoe-multimodal-cgqn-v1':
    model = net.APoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'apoe-multimodal-cgqn-v2':
    model = net.APoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'apoe-multimodal-cgqn-v3':
    model = net.APoEMultimodalCGQN_v3(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'apoe-multimodal-cgqn-v4':
    model = net.APoEMultimodalCGQN_v4(dims=dataset_info['dims'], num_layers=1).to(device)

elif opt.model == 'apoe-multimodal-ml-cgqn-v1':
    model = net.APoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=2).to(device)
elif opt.model == 'apoe-multimodal-ml-cgqn-v2':
    model = net.APoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=2).to(device)

elif opt.model == 'conv-apoe-multimodal-cgqn-v1':
    model = net.ConvAPoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'conv-apoe-multimodal-cgqn-v2':
    model = net.ConvAPoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'conv-apoe-multimodal-cgqn-v3':
    model = net.ConvAPoEMultimodalCGQN_v3(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'conv-apoe-multimodal-cgqn-v4':
    model = net.ConvAPoEMultimodalCGQN_v4(dims=dataset_info['dims'], num_layers=1).to(device)

elif opt.model == 'conv-apoe-multimodal-ml-cgqn-v1':
    model = net.ConvAPoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=2).to(device)
elif opt.model == 'conv-apoe-multimodal-ml-cgqn-v2':
    model = net.ConvAPoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=2).to(device)

else:
    raise NotImplementedError('unknown model: {}'.format(opt.model))
logging(str(model), path=opt.new_path)

with open(os.path.join(opt.path, 'model.pt'), 'rb') as f:
    pretrained_model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
    model.load_state_dict(pretrained_model.state_dict())
    del pretrained_model


# msc
def get_batch_size(eval_target):
    mod_batch_sizes = []
    for i in range(num_modalities):
        mod_batch_sizes += [sum([target[i*2].size(0) for target in eval_target if target[i*2] is not None])]
    batch_size = sum(mod_batch_sizes)
    return batch_size, mod_batch_sizes

def get_dim_size(eval_target, is_grayscale=False):
    num_episodes = len(eval_target)

    dim_per_eps = []
    for i in range(num_episodes):
        dim = 0
        for j in range(num_modalities):
            channels, height, width, nc_query, mtype = dataset_info['dims'][j]
            if eval_target[i][j*2] is not None:
                _dim = eval_target[i][j*2].numel()
                if mtype == 'image' and is_grayscale:
                    _dim = _dim // channels
                dim += _dim
        dim_per_eps += [dim]

    return dim_per_eps

def get_str_from_mask(mask):
    if type(mask) != type([]):
        mask = list(mask)
    return ''.join(['{}']*num_modalities).format(*mask)

def build_indices_for_masks(mask_combinations):
    mask_indices = {}
    for idx, _mask in enumerate(mask_combinations):
        mask = list(_mask)
        key = get_str_from_mask(mask)
        assert not key in mask_indices
        mask_indices[key] = idx
    return mask_indices
mask_indices = build_indices_for_masks(mask_combinations)

def get_idx_from_mask(mask):
    key = get_str_from_mask(mask)
    return mask_indices[key]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# evaluate
def evaluate(eval_loader, name='val'):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    start_time = time.time()
    #NUM_Q_SAMPLES = opt.num_q_samples
    NUM_Z_SAMPLES = opt.num_z_samples
    NUM_ITERS = min(opt.num_iters, len(eval_loader))
    MOD_STEP = opt.mod_step #5 #3
    NUM_MODS = sorted(list(set([1] + [n_mods for n_mods in range(2, num_modalities+1, MOD_STEP)] + [num_modalities])))
    NUM_CONTEXTS = [0, 1, 2, 3, 4, 5]
    NUM_TARGET  = 5
    MASK_STEP = opt.mask_step #10 #5

    all_masks = []
    for n_mods in NUM_MODS:
        masks = get_masks(num_modalities, min_modes=n_mods, max_modes=n_mods)
        masks = list(set(masks[::MASK_STEP]+[masks[-1]])) 
        all_masks += masks
    m_indices = dict(zip([get_str_from_mask(mask) for mask in all_masks], [i for i in range(len(all_masks))]))

    logging('num mods : {}'.format(NUM_MODS), path=opt.path)
    logging('num ctxs : {}'.format(NUM_CONTEXTS), path=opt.path)
    logging('num tgt  : {}'.format(NUM_TARGET), path=opt.path)
    logging('mask step: {}'.format(MASK_STEP), path=opt.path)
    logging('masks    : {}'.format(m_indices), path=opt.path)

    total_avg_batch_sizes_per_nmod_nctx = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(num_modalities)]
    total_avg_acc1_per_nmod_nctx        = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(num_modalities)]
    total_avg_acc5_per_nmod_nctx        = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(num_modalities)]
    total_batch_sizes_per_nmod_nctx = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(len(all_masks))]
    total_acc1_per_nmod_nctx        = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(len(all_masks))]
    total_acc5_per_nmod_nctx        = [[0 for i in range(len(NUM_CONTEXTS))] for j in range(len(all_masks))]

    with torch.no_grad():
        for batch_idx, (eval_info, eval_context, eval_target) in enumerate(eval_loader):
            # init batch
            eval_context = batch_to_device(eval_context, device)
            eval_target  = batch_to_device(eval_target, device)
            eval_all = merge_two_batch(eval_context, eval_target)
            num_episodes = len(eval_context)

            # select target
            _new_eval_target = []
            for target in eval_all:
                _target = tuple([target[i][-NUM_TARGET:] if target[i] is not None else None for i in range(len(target))])
                _new_eval_target += [_target]
            eval_target = _new_eval_target

            # forward
            for n_mods in NUM_MODS:
                masks = get_masks(num_modalities, min_modes=n_mods, max_modes=n_mods)
                masks = list(set(masks[::MASK_STEP]+[masks[-1]])) 
                for mask in masks:
                    n_mods = sum(mask)
                    avg_m_idx = n_mods-1
                    m_idx = m_indices[get_str_from_mask(mask)]
                    for c_idx, num_context in enumerate(NUM_CONTEXTS):
                        # select context
                        _new_eval_context = []
                        for context in eval_all:
                            _context = tuple([context[i][:num_context] if context[i] is not None and num_context > 0 and mask[i//2] else None for i in range(len(context))])
                            _new_eval_context += [_context]
                        eval_context = _new_eval_context

                        # get labels
                        eval_label = torch.Tensor([i for i in range(num_episodes)]).long().to(device)

                        # infer
                        logprobs_per_batch = []
                        for i_ep in range(num_episodes):
                            new_eval_target = [eval_target[i_ep]]*num_episodes

                            # get dim size per episode
                            dim_per_eps = get_dim_size(new_eval_target, is_grayscale=opt.grayscale)

                            # forward
                            logprobs = []
                            for j in range(NUM_Z_SAMPLES):
                                # forward
                                _, _, logprob, info = model.predict(eval_context, new_eval_target, is_grayscale=opt.grayscale, use_uint8=False)

                                # append to loss_logprobs
                                logprobs += [logprob.unsqueeze(1)]

                            # concat
                            logprobs = torch.cat(logprobs, dim=1)

                            # get logprob
                            logprobs = logprob_logsumexp(logprobs).detach()

                            # get logprob per dimension
                            for i in range(num_episodes):
                                logprobs[i:i+1] /= float(dim_per_eps[i])

                            # append
                            logprobs_per_batch += [logprobs.unsqueeze(1)]

                        # concat
                        logprobs_per_batch = torch.cat(logprobs_per_batch, dim=1)

                        # get acc
                        acc1, acc5 = accuracy(logprobs_per_batch, eval_label, topk=(1, 5))
                        cur_acc1 = acc1[0].item()
                        cur_acc5 = acc5[0].item()
                        total_avg_acc1_per_nmod_nctx[avg_m_idx][c_idx]        += cur_acc1 * num_episodes
                        total_avg_acc5_per_nmod_nctx[avg_m_idx][c_idx]        += cur_acc5 * num_episodes
                        total_avg_batch_sizes_per_nmod_nctx[avg_m_idx][c_idx] += num_episodes
                        total_acc1_per_nmod_nctx[m_idx][c_idx]        += cur_acc1 * num_episodes
                        total_acc5_per_nmod_nctx[m_idx][c_idx]        += cur_acc5 * num_episodes
                        total_batch_sizes_per_nmod_nctx[m_idx][c_idx] += num_episodes

            # plot
            if (batch_idx+1) % opt.vis_interval == 0 or (batch_idx+1) == len(eval_loader):
                elapsed = time.time() - start_time
                start_time = time.time()

                # print
                logging('| {} '
                        '| {:5d}/{:5d} '
                        '| sec/step {:5.2f} '
                        '| acc (top1) {:.3f} '
                        '| acc (top5) {:.3f} '
                        .format(
                        name,
                        batch_idx+1, len(eval_loader),
                        elapsed / opt.vis_interval,
                        cur_acc1,
                        cur_acc5,
                        ),
                        path=opt.path)

            if (batch_idx+1) == NUM_ITERS:
                break

    # print
    logging(''.join(['masks V / # of context > '] + ['  {:4d}'.format(num_context) for num_context in NUM_CONTEXTS]), path=opt.new_path)
    logging('='*17 + ' acc1 ' + '='*17 + ' | ' + '='*17 + ' acc5 ' + '='*17, path=opt.new_path)
    for mask in all_masks:
        mask_str = get_str_from_mask(mask)
        m_idx = m_indices[mask_str]
        txt = ' {} |'.format(mask_str)
        for c_idx, num_context in enumerate(NUM_CONTEXTS):
            total_batch_size = total_batch_sizes_per_nmod_nctx[m_idx][c_idx]
            total_acc1 = total_acc1_per_nmod_nctx[m_idx][c_idx] / total_batch_size
            writer.add_scalar('mask{}/{}/acc1'.format(mask_str, name), total_acc1, num_context)
            txt += '  {:3.1f}'.format(total_acc1)
        txt += ' | '
        for c_idx, num_context in enumerate(NUM_CONTEXTS):
            total_batch_size = total_batch_sizes_per_nmod_nctx[m_idx][c_idx]
            total_acc5 = total_acc5_per_nmod_nctx[m_idx][c_idx] / total_batch_size
            writer.add_scalar('mask{}/{}/acc5'.format(mask_str, name), total_acc5, num_context)
            txt += '  {:3.1f}'.format(total_acc5)
        logging(txt, path=opt.new_path)

    # print
    logging('', path=opt.new_path)
    logging('', path=opt.new_path)
    logging(''.join(['# of mods V / # of context > '] + ['  {:4d}'.format(num_context) for num_context in NUM_CONTEXTS]), path=opt.new_path)
    logging('='*17 + ' acc1 ' + '='*17 + ' | ' + '='*17 + ' acc5 ' + '='*17, path=opt.new_path)
    for n_mods in NUM_MODS:
        avg_m_idx = n_mods-1
        txt = ' {} |'.format(n_mods)
        for c_idx, num_context in enumerate(NUM_CONTEXTS):
            total_avg_batch_size = total_avg_batch_sizes_per_nmod_nctx[avg_m_idx][c_idx]
            total_avg_acc1 = total_avg_acc1_per_nmod_nctx[avg_m_idx][c_idx] / total_avg_batch_size
            writer.add_scalar('M{}/{}/acc1'.format(n_mods, name), total_avg_acc1, num_context)
            writer.add_scalar('C{}/{}/acc1'.format(num_context, name), total_avg_acc1, n_mods)
            txt += '  {:3.1f}'.format(total_avg_acc1)
        txt += ' | '
        for c_idx, num_context in enumerate(NUM_CONTEXTS):
            total_avg_batch_size = total_avg_batch_sizes_per_nmod_nctx[avg_m_idx][c_idx]
            total_avg_acc5 = total_avg_acc5_per_nmod_nctx[avg_m_idx][c_idx] / total_avg_batch_size
            writer.add_scalar('M{}/{}/acc5'.format(n_mods, name), total_avg_acc5, num_context)
            writer.add_scalar('C{}/{}/acc5'.format(num_context, name), total_avg_acc5, n_mods)
            txt += '  {:3.1f}'.format(total_avg_acc5)
        logging(txt, path=opt.new_path)

    return total_acc1 / total_batch_size, total_acc5 / total_batch_size


# Run on test data
logging('', path=opt.new_path)
logging('', path=opt.new_path)
test_acc1, test_acc5   = evaluate(test_loader,  name='test')
logging('=' * 89, path=opt.new_path)
logging('| End of classification '
        '| test acc1 {:5.8f} '
        '| test acc5 {:5.8f} '.format(
        test_acc1,
        test_acc5),
        path=opt.new_path)
