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
from utils import binary_trim_context_target
from utils import sample_queries, sample_hand_queries, sample_random_queries, sample_random_hand_queries
from utils import get_visualization_image_data, get_visualization_haptic_data, get_combined_visualization_image_data

import matplotlib.pyplot as plt
import seaborn as sns


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

# training
parser.add_argument('--train-batch-size', type=int, default=128,
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=1,
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
parser.add_argument('--uint8', action='store_true', default=False,
                    help='eval image data in uint8')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cache', default='eval', help='path to cache')
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
train_loader, val_loader, _, test_loader, dataset_info = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda)
num_modalities = dataset_info['num_modalities']


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


# evaluate
def evaluate(eval_loader, test=False):
    # Turn on evaluation mode which disables dropout.
    name='test' if test else 'val'
    model.eval()
    start_time = time.time()
    NUM_TEST = 5
    NUM_CONTEXTS = [0, 1, 5, 10]
    indices_NUM_CONTEXTS = {0:0, 1:1, 5:2, 10:3}
    assert (NUM_TEST + NUM_CONTEXTS[-1]) <= dataset_info['nviews']
    NUM_Q_SAMPLES = opt.num_q_samples
    NUM_Z_SAMPLES = opt.num_z_samples
    NUM_ITERS = min(opt.num_iters, len(eval_loader))
    total_loss = 0.
    total_batch_size = 0
    total_mod_batch_sizes_per_sources = [[
        [0 for i in range(num_modalities)]
        for num_context in NUM_CONTEXTS]
        for num_context in NUM_CONTEXTS]
    total_mod_logprobs_per_sources = [[
        [0 for i in range(num_modalities)]
        for num_context in NUM_CONTEXTS]
        for num_context in NUM_CONTEXTS]

    with torch.no_grad():
        for i_query in range(NUM_Q_SAMPLES):
            for batch_idx, (eval_info, eval_context, eval_target) in enumerate(eval_loader):
                # init batch
                eval_context = batch_to_device(eval_context, device)
                eval_target  = batch_to_device(eval_target, device)

                # get merged context / target
                eval_all = merge_two_batch(eval_context, eval_target)
                num_episodes = len(eval_all)

                ''' temporary '''
                def new_eval(img_num_context=0, hpt_num_context=0):
                    # get context
                    new_eval_context, new_eval_target = binary_trim_context_target(eval_all, img_num_context=img_num_context, hpt_num_context=hpt_num_context, num_modalities=num_modalities)
                    if sum([int(new_eval_target[0][j*2] is None) for j in range(num_modalities)]) == num_modalities:
                        return

                    # select test
                    _new_eval_target = []
                    for target in new_eval_target:
                        _target = tuple([target[i][-NUM_TEST:] for i in range(len(target))])
                        _new_eval_target += [_target]
                    new_eval_target = _new_eval_target

                    # get batch size
                    _, mod_batch_sizes = get_batch_size(new_eval_target)

                    # loss
                    loss_mod_logprobs = [None]*num_modalities
                    for i in range(num_modalities):
                        if new_eval_target[i*2] is None:
                            pass
                        else:
                            newnew_eval_target = [
                                    tuple([target[j] if j//2 == i else None for j in range(num_modalities*2)])
                                    for target in new_eval_target
                                    ]

                            # get batch size
                            batch_size, _ = get_batch_size(newnew_eval_target)
                            assert batch_size == mod_batch_sizes[i]

                            # get dim size per episode
                            dim_per_eps = get_dim_size(newnew_eval_target, is_grayscale=opt.grayscale)

                            # forward
                            logprobs = []
                            for j in range(NUM_Z_SAMPLES):
                                # forward
                                _, _, logprob, info = model.predict(new_eval_context, newnew_eval_target, is_grayscale=opt.grayscale, use_uint8=opt.uint8)

                                # append to loss_logprobs
                                logprobs += [logprob.unsqueeze(1)]

                            # concat
                            logprobs = torch.cat(logprobs, dim=1)

                            # get logprob
                            _logprobs_max, _ = torch.max(logprobs, dim=1, keepdim=True)
                            _logprobs = logprobs - _logprobs_max # w - \hat(w)
                            _logprobs = torch.log(torch.sum(_logprobs.exp(), dim=1, keepdim=True)) # log sum(exp(w - \hat(w)))
                            logprobs = -math.log(float(NUM_Z_SAMPLES)) + _logprobs_max + _logprobs # log(1/NUM_Z_SAMPLES) + w + log sum(exp(w - \hat(w)))

                            # get logprob per dimension
                            for j in range(num_episodes):
                                logprobs[j:j+1] /= float(dim_per_eps[j])

                            # add to total_mod_logprobs
                            loss_mod_logprobs[i] = torch.sum(logprobs)

                    # add to total_loss
                    for i in range(num_modalities):
                        total_mod_logprobs_per_sources[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i]    += loss_mod_logprobs[i].item() if loss_mod_logprobs[i] is not None else 0
                        total_mod_batch_sizes_per_sources[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i] += mod_batch_sizes[i]

                # run new_eval
                for hpt_num_context in NUM_CONTEXTS:
                    for img_num_context in NUM_CONTEXTS:
                        new_eval(img_num_context, hpt_num_context)

                if (batch_idx+1) % opt.vis_interval == 0:
                    elapsed = time.time() - start_time
                    print('[', i_query+1, '/', NUM_Q_SAMPLES, ']', '  ', batch_idx+1, '/', NUM_ITERS, 'elapsed: {:.3f} ms'.format(elapsed*1000/opt.vis_interval))
                    start_time = time.time()

                if (batch_idx+1) == NUM_ITERS:
                    break

    # add to total_loss
    for hpt_num_context in NUM_CONTEXTS:
        for img_num_context in NUM_CONTEXTS:
            for i, (channels, height, width, nc_query, mtype) in enumerate(dataset_info['dims']):
                batch_size = total_mod_batch_sizes_per_sources[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i]
                if mtype == 'image' and opt.grayscale:
                    dim = 1*height*width
                else:
                    dim = channels*height*width
                if batch_size > 0:
                    total_mod_logprobs_per_sources[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i] /= float(batch_size*dim)
                else:
                    total_mod_logprobs_per_sources[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i] = None

    # print
    for loss_func, loss_name in zip(
            [total_mod_logprobs_per_sources],
            ['logprob']):
        i = 0
        logging('', path=opt.new_path)
        logging('', path=opt.new_path)
        logging('', path=opt.new_path)
        logging('--------------------', path=opt.new_path)
        logging('({}) predict = img (per pixel/dim)'.format(loss_name), path=opt.new_path)
        logging(''.join(['hpt: V, img: > '] + ['  {:4d}'.format(img_num_context) for img_num_context in NUM_CONTEXTS]), path=opt.new_path)
        for hpt_num_context in NUM_CONTEXTS:
            txt = '{:4d}'.format(hpt_num_context)
            for img_num_context in NUM_CONTEXTS:
                loss = loss_func[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i]
                writer.add_scalar('{}/pred_img/nch{}/'.format(loss_name, hpt_num_context), loss if loss is not None else -float("inf"), img_num_context)
                writer.add_scalar('{}/pred_img/nci{}/'.format(loss_name, img_num_context), loss if loss is not None else -float("inf"), hpt_num_context)
                txt += '  {:5.8f}'.format(loss if loss is not None else -float("inf"))
            logging(txt, path=opt.new_path)
        logging('--------------------', path=opt.new_path)

        i = 1 
        logging('', path=opt.new_path)
        logging('--------------------', path=opt.new_path)
        logging('({}) predict = hpt (per pixel/dim)'.format(loss_name), path=opt.new_path)
        logging(''.join(['hpt: V, img: > '] + ['  {:4d}'.format(img_num_context) for img_num_context in NUM_CONTEXTS]), path=opt.new_path)
        for hpt_num_context in NUM_CONTEXTS:
            txt = '{:4d}'.format(hpt_num_context)
            for img_num_context in NUM_CONTEXTS:
                loss = loss_func[indices_NUM_CONTEXTS[img_num_context]][indices_NUM_CONTEXTS[hpt_num_context]][i]
                writer.add_scalar('{}/pred_hpt/nch{}/'.format(loss_name, hpt_num_context), loss if loss is not None else -float("inf"), img_num_context)
                writer.add_scalar('{}/pred_hpt/nci{}/'.format(loss_name, img_num_context), loss if loss is not None else -float("inf"), hpt_num_context)
                txt += '  {:5.8f}'.format(loss if loss is not None else -float("inf"))
            logging(txt, path=opt.new_path)
        logging('--------------------', path=opt.new_path)

    return total_mod_logprobs_per_sources


# Run on val data
val_logprob = evaluate(val_loader, test=False)
logging('=' * 89, path=opt.new_path)

## Run on test data
#test_logprob = evaluate(test_loader, test=True)
#logging('=' * 89, path=opt.new_path)
