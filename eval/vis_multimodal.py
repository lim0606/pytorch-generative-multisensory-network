# import
import os
import argparse
import time
import math

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms

import datasets as dset
import models as net

from utils import get_time, logging, get_lrs
from utils import get_image_from_values, get_plot, get_grid_image
from utils import batch_to_device, merge_two_batch, get_masks#, trim_context_target
from utils.gqn_tmp import new_trim_context_target as trim_context_target
from utils import sample_queries, sample_hand_queries, sample_random_queries, sample_random_hand_queries
from utils import get_visualization_image_data, get_visualization_haptic_data, get_combined_visualization_image_data, get_combined_visualization_haptic_data
from utils import convert_npimage_torchimage

from PIL import Image


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
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for test (default: 10)')
parser.add_argument('--num-samples', type=int, default=1, #5
                    help='number of sampled images')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of iters used for evaluation; thus, the number of evaluated data = num_iters * eval_batch_size')

# eval mode
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='eval image data in grayscale')
parser.add_argument('--n-context', type=int, action='append', default=[],
                    help='number of contexts')
parser.add_argument('--n-mods', type=int, action='append', default=[],
                    help='number of modes')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cache', default='vis', help='path to cache')
parser.add_argument('--path', type=str, default='./model.pt',
                    help='model file to load')

# parse arguments
opt = parser.parse_args()
opt.new_path = os.path.join(opt.cache, opt.path)

# check condition
assert opt.eval_batch_size == 1

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if opt.cuda else "cpu")

# print args
logging(str(opt), log_=False)

# init tensorboard
writer = SummaryWriter(opt.new_path)
os.system('mkdir -p {}'.format(os.path.join(opt.new_path, 'val')))
os.system('mkdir -p {}'.format(os.path.join(opt.new_path, 'test')))

# init dataset
train_loader, val_loader, _, test_loader, dataset_info = dset2.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda)
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
num_classes = 728

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
logging(str(model), log_=False)

with open(os.path.join(opt.path, 'model.pt'), 'rb') as f:
    pretrained_model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
    model.load_state_dict(pretrained_model.state_dict())
    del pretrained_model


# msc
def get_transform(img_size=64, do_crop=False):
    trfms = []
    if do_crop:
        trfms += [transforms.CenterCrop(132)]
    trfms += [transforms.Resize(img_size), #, interpolation=Image.NEAREST),
              transforms.ToTensor()]
    return transforms.Compose(trfms)

def load_images(images, transform):
    new_images = []
    for i in range(len(images)):
        # load image
        image = images[i]
        image = Image.fromarray(image, mode='RGB')
        image = transform(image)
        new_images += [image.unsqueeze(0)]
    new_images = torch.cat(new_images, dim=0)
    return new_images
def get_batch_size(eval_target):
    mod_batch_sizes = []
    for i in range(num_modalities):
        mod_batch_sizes += [sum([target[i*2].size(0) for target in eval_target if target[i*2] is not None])]
    batch_size = sum(mod_batch_sizes)
    return batch_size, mod_batch_sizes

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

def get_visualization_queries_with_predefined_dist(num_episodes, device, img_queries=None):
    # generate image with shuffled queries and random queries
    if img_queries is None:
        img_queries = sample_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False).to(device)
    else:
        img_queries = img_queries.to(device)
    num_img_queries = img_queries.size(0)
    img_queries = [img_queries for i in range(num_episodes)]
    return img_queries, num_img_queries

def get_visualization_queries_from_data(idx, eval_target, num_episodes, num_hpt_queries=5):
    queries = [eval_target[i][idx*2+1][-num_hpt_queries:] for i in range(num_episodes)]
    return queries, num_hpt_queries

def get_visualization_queries_for_haptic(num_episodes, device):
    hpt_queries = sample_hand_queries(nrow=4, ncol=8).to(device)
    num_hpt_queries = hpt_queries.size(0)
    set_indices = [np.random.permutation(num_hpt_queries)[:num_hpt_queries] % hpt_queries.size(0) for i in range(num_episodes)]
    shuffled_hpt_queries = [torch.cat([hpt_queries[i:i+1] for i in indices], dim=0) for indices in set_indices]
    return shuffled_hpt_queries, num_hpt_queries

def get_queries(eval_target, device, num_hpt_queries=5, img_queries=None):
    num_episodes = len(eval_target)

    # init queries
    mod_queries, num_mod_queries = [], []
    for idx, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
        # get queries
        if mtype == 'image':
            # image queries
            _mod_queries, _num_mod_queries = get_visualization_queries_with_predefined_dist(num_episodes, device, img_queries)
        elif mtype == 'haptic':
            # haptic queries
            _mod_queries, _num_mod_queries = get_visualization_queries_from_data(idx, eval_target, num_episodes, num_hpt_queries)

        # append to list
        mod_queries += [_mod_queries]
        num_mod_queries += [_num_mod_queries]

    return mod_queries, num_mod_queries


# evaluate
def evaluate(eval_loader, test=False):
    # Turn on evaluation mode which disables dropout.
    name='test' if test else 'val'
    model.eval()
    transform = get_transform()
    NUM_ITERS = min(opt.num_iters, len(eval_loader))
    NUM_TEST = 5
    NUM_CONTEXTS = sorted([nc for nc in opt.n_context])
    if len(NUM_CONTEXTS) == 0:
        NUM_CONTEXTS = [0, 1, 5, 10] #[0, 1, 5, 10, 15]
    assert (NUM_TEST + NUM_CONTEXTS[-1]) <= dataset_info['nviews']
    NUM_MODS = sorted([n_mods for n_mods in opt.n_mods])
    if len(NUM_MODS) == 0:
        NUM_MODS = [n_mod for n_mod in range(1, num_modalities+1)]
    assert NUM_MODS[-1] <= num_modalities
    assert NUM_MODS[0] > 0 
    all_masks = []
    for n_mods in NUM_MODS:
        masks = get_masks(num_modalities, min_modes=n_mods, max_modes=n_mods)
        all_masks += masks

    logging('num mods : {}'.format(NUM_MODS), path=opt.path)
    logging('num ctxs : {}'.format(NUM_CONTEXTS), path=opt.path)
    logging('num tgt  : {}'.format(NUM_TEST), path=opt.path)
    logging('masks    : {}'.format(all_masks), path=opt.path)

    hpt_tgt_gen = {}
    avg_diffs = {}
    num_datas = {}
    with torch.no_grad():
        for i_sample in range(1, opt.num_samples+1):
            did_plot = [False]*num_classes
            for batch_idx, (eval_info, eval_context, eval_target) in enumerate(eval_loader):
                # init batch
                eval_context = batch_to_device(eval_context, device)
                eval_target  = batch_to_device(eval_target, device)
                eval_all = merge_two_batch(eval_context, eval_target)
                num_episodes = len(eval_context)

                # get img_queries
                img_queries = torch.from_numpy(np.array(eval_info[0]['add_cameras'])).float()

                # get true_images and hand_images
                true_images = load_images(eval_info[0]['add_images'], transform)
                _true_images = get_grid_image(true_images, 16, 3, 64, 64, nrow=4, pad_value=0)
                hand_images = load_images(eval_info[0]['hand_images'], transform)
                _hand_images = get_grid_image(hand_images, 15, 3, 64, 64, nrow=15, pad_value=0)
                _data_images = []
                for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                    if mtype == 'image':
                        _data_images += [eval_all[0][idx*2]]
                _data_images = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], _data_images, dataset_info['nviews'], min(4, num_episodes), nrow=15, pad_value=0)[0]

                ''' temporary '''
                assert len(eval_context) == 1
                assert len(eval_target) == 1
                cls = eval_info[0]['class']

                ''' per class '''
                # visualize per class
                if not did_plot[cls]:
                    # change flag
                    did_plot[cls] = True

                    # draw true_images and hand_images
                    writer.add_image('{}/gt-img-cls{}-i{}'.format(name, cls, i_sample), _true_images, 0)
                    writer.add_image('{}/hand-img-cls{}-i{}'.format(name, cls, i_sample), _hand_images, 0)
                    writer.add_image('{}/data-img-cls{}-i{}'.format(name, cls, i_sample), _data_images, 0)
                    for num_context in NUM_CONTEXTS:
                        _hand_images = get_grid_image(hand_images[:num_context], num_context, 3, 64, 64, nrow=5, pad_value=0)
                        writer.add_image('{}/ctx-hand-img-cls{}-i{}-nc{}'.format(name, cls, i_sample, num_context), _hand_images, 0)

                    def draw_img_gen(mask, num_context=0):
                        # get mask index
                        m_idx = sum(mask)-1

                        # get context
                        new_eval_context, new_eval_target = trim_context_target(eval_all, num_context=num_context, mask=mask, num_modalities=num_modalities)
                        if sum([int(new_eval_target[0][j*2] is None) for j in range(num_modalities)]) == num_modalities:
                            return

                        # select test
                        _new_eval_target = []
                        for i in range(num_episodes):
                            _target = []
                            for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                                data, query = new_eval_target[i][idx*2], new_eval_target[i][idx*2+1]
                                if mtype == 'haptic':
                                    _target += [data[-NUM_TEST:]  if data is not None else None]
                                    _target += [query[-NUM_TEST:] if data is not None else None]
                                else:
                                    _target += [data]
                                    _target += [query]
                            _new_eval_target += [tuple(_target)]
                        new_eval_target = _new_eval_target

                        # get batch size
                        batch_size, mod_batch_sizes = get_batch_size(new_eval_target)

                        # get queries
                        mod_queries, num_mod_queries = get_queries(new_eval_target, device, num_hpt_queries=NUM_TEST, img_queries=img_queries)

                        # forward
                        outputs, _, _, _ = model(new_eval_context, new_eval_target, is_grayscale=opt.grayscale)

                        # generate
                        gens, _ = model.generate(new_eval_context, tuple(mod_queries), is_grayscale=opt.grayscale)

                        # visualize
                        img_ctxs, img_tgts, img_outputs, img_gens = [], [], [], []
                        hpt_ctxs, hpt_tgts, hpt_outputs, hpt_gens = [], [], [], []
                        for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                            # get output and gen
                            output = outputs[idx]
                            gen = gens[idx]
                            _num_mod_queries = num_mod_queries[idx]

                            # visualize
                            if mtype == 'image':
                                # grayscale
                                if opt.grayscale:
                                    if output.size(0) > 0:
                                        output = output.expand(output.size(0), nchannels, nheight, nwidth)
                                    gen = gen.expand(gen.size(0), nchannels, nheight, nwidth)

                                # get ctx, tgt
                                if num_context > 0 and mask[idx]:
                                    sz = new_eval_context[0][idx*2].size()[1:]
                                    ctx = torch.cat([
                                        new_eval_context[0][idx*2],
                                        gen.new_zeros(dataset_info['nviews']-num_context, *sz)], dim=0)
                                    num_target = new_eval_target[0][idx*2].size(0) if new_eval_target[0][idx*2] is not None else 0
                                    assert num_target == output.size(0)
                                    if num_target > 0:
                                        tgt = torch.cat([
                                            gen.new_zeros(dataset_info['nviews']-num_target, *sz),
                                            new_eval_target[0][idx*2],
                                            ], dim=0)
                                        output = torch.cat([
                                            gen.new_zeros(dataset_info['nviews']-num_target, *sz),
                                            output,
                                            ], dim=0)
                                    else:
                                        tgt = gen.new_zeros(dataset_info['nviews']*num_episodes, *sz)
                                        output = gen.new_zeros(dataset_info['nviews']*num_episodes, *sz)
                                else:
                                    ctx = gen.new_zeros(dataset_info['nviews']*num_episodes, nchannels, nheight, nwidth)
                                    tgt = new_eval_target[0][idx*2]

                                # append to list
                                img_gens += [gen]
                                img_outputs += [output]
                                img_ctxs += [ctx]
                                img_tgts += [tgt]
                                num_img_queries = _num_mod_queries
                            elif mtype == 'haptic':
                                ctx = new_eval_context[0][idx*2]
                                tgt = new_eval_target[0][idx*2]

                                # append to list
                                hpt_gens += [gen]
                                hpt_outputs += [output]
                                hpt_ctxs += [ctx]
                                hpt_tgts += [tgt]
                                num_hpt_queries = _num_mod_queries
                            else:
                                raise NotImplementedError

                        # combine haptic
                        if not get_str_from_mask(mask) in hpt_tgt_gen:
                            hpt_tgt_gen[get_str_from_mask(mask)] = {}
                            avg_diffs[get_str_from_mask(mask)] = np.zeros(len(NUM_CONTEXTS))
                            num_datas[get_str_from_mask(mask)] = 0
                        hpt_tgts = torch.cat(hpt_tgts, dim=1)
                        hpt_gens = torch.cat(hpt_gens, dim=1)
                        hpt_tgt_gen[get_str_from_mask(mask)][num_context] = (hpt_tgts, hpt_gens)

                        # visualize combined image
                        xgs = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_gens, num_img_queries, min(4, num_episodes), nrow=4, pad_value=0)
                        xos = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_outputs, dataset_info['nviews'], min(4, num_episodes), nrow=4, pad_value=0)
                        xcs = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_ctxs, dataset_info['nviews'], min(4, num_episodes), nrow=4, pad_value=0)
                        _xcs = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_ctxs, num_context, min(4, num_episodes), nrow=5, pad_value=0)
                        xts = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_tgts, dataset_info['nviews'], min(4, num_episodes), nrow=4, pad_value=0)
                        for i, (xc, xt, xo, xg) in enumerate(zip(xcs, xts, xos, xgs)):
                            writer.add_image('{}/ctx-cls{}-M{}-mask{}-i{}-nc{}/img'.format(name, cls, m_idx+1, get_str_from_mask(mask), i, num_context), _xcs[i], 0)
                            writer.add_image('{}/gen-cls{}-M{}-mask{}-i{}-nc{}/img'.format(name, cls, m_idx+1, get_str_from_mask(mask), i, num_context), xg, 0)
                            x = torch.cat([xc, xt, xo, xg], dim=2)
                            writer.add_image('{}/ctx-tgt-rec-gen-cls{}-M{}-mask{}-i{}-nc{}/img'.format(name, cls, m_idx+1, get_str_from_mask(mask), i, num_context), x, 0)

                    # run vis
                    for mask in all_masks:
                        for num_context in NUM_CONTEXTS:
                            draw_img_gen(mask=mask, num_context=num_context)

                        # visualize combined haptic
                        m_idx = sum(mask)-1 # get mask index
                        xs, diffs = get_combined_visualization_haptic_data(hpt_tgt_gen[get_str_from_mask(mask)], title='mask: {}'.format(get_str_from_mask(mask)))
                        _diffs = [np.mean([diffs[i][j] for i in range(len(diffs))]) for j in range(len(NUM_CONTEXTS))]
                        num_datas[get_str_from_mask(mask)] += 1
                        for j, diff in enumerate(_diffs):
                            avg_diffs[get_str_from_mask(mask)][j:j+1] += diff
                            num_context = NUM_CONTEXTS[j]
                            writer.add_scalar('{}/diff-cls{}-M{}-mask{}-all/hpt'.format(name, cls, m_idx+1, get_str_from_mask(mask)), diff, num_context)

                        for i, x in enumerate(xs):
                            writer.add_image('{}/tgt-gen-cls{}-M{}-mask{}-i{}/hpt'.format(name, cls, m_idx+1, get_str_from_mask(mask), i), convert_npimage_torchimage(x), 0)

                            for j, diff in enumerate(diffs[i]):
                                num_context = NUM_CONTEXTS[j]
                                writer.add_scalar('{}/diff-cls{}-M{}-mask{}-i{}/hpt'.format(name, cls, m_idx+1, get_str_from_mask(mask), i), diff, num_context)

                if (batch_idx+1) % 1 == 0:
                    print(batch_idx+1, '/', NUM_ITERS, ' [', len(eval_loader), ']')
                if (batch_idx+1) == NUM_ITERS:
                    break

    return

# Run on val data
evaluate(val_loader, test=False)
logging('=' * 89, log_=False)
logging('| End of visualization ', log_=False)
logging('=' * 89, log_=False)
