# import
import os
import argparse
import time
import math
from PIL import Image

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
from utils import batch_to_device, merge_two_batch, trim_context_target
from utils import sample_queries, sample_hand_queries, sample_random_queries, sample_random_hand_queries
from utils import get_visualization_image_data, get_visualization_haptic_data, get_combined_visualization_image_data

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
parser.add_argument('--num-samples', type=int, default=5,
                    help='number of sampled images')

# eval mode
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='eval image data in grayscale')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cache', default='vis', help='path to cache')
parser.add_argument('--path', type=str, default='./model.pt',
                    help='model file to load')

# parse arguments
opt = parser.parse_args()

# check condition
assert opt.eval_batch_size == 1

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if opt.cuda else "cpu")

# print args
logging(str(opt), log_=False)

# init tensorboard
writer = SummaryWriter(os.path.join(opt.cache, opt.path))
os.system('mkdir -p {}'.format(os.path.join(opt.cache, opt.path, 'val')))
os.system('mkdir -p {}'.format(os.path.join(opt.cache, opt.path, 'test')))

# init dataset
train_loader, val_loader, _, test_loader, dataset_info = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda)
num_modalities = dataset_info['num_modalities']
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

def convert_npimage_torchimage(images):
    # image: c x h x w
    #return torch.transpose(torch.transpose(torch.from_numpy(image.astype(np.float)), 0, 2), 1, 2)
    # images: b x c x h x w
    return torch.transpose(torch.transpose(torch.from_numpy(images.astype(np.float)), 1, 3), 2, 3)

def get_batch_size(eval_target):
    mod_batch_sizes = []
    for i in range(num_modalities):
        mod_batch_sizes += [sum([target[i*2].size(0) for target in eval_target if target[i*2] is not None])]
    batch_size = sum(mod_batch_sizes)
    return batch_size, mod_batch_sizes

def get_visualization_queries_with_predefined_dist(num_episodes, device, img_queries=None):
    # generate image with shuffled queries and random queries
    if img_queries is None:
        img_queries = sample_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False).to(device)
    else:
        img_queries = img_queries.to(device)
    num_img_queries = img_queries.size(0)
    img_queries = [img_queries for i in range(num_episodes)]
    return img_queries, num_img_queries

def get_visualization_queries_for_haptic(num_episodes, device):
    hpt_queries = sample_hand_queries(nrow=4, ncol=8).to(device)
    num_hpt_queries = hpt_queries.size(0)
    set_indices = [np.random.permutation(num_hpt_queries)[:num_hpt_queries] % hpt_queries.size(0) for i in range(num_episodes)]
    shuffled_hpt_queries = [torch.cat([hpt_queries[i:i+1] for i in indices], dim=0) for indices in set_indices]
    return shuffled_hpt_queries, num_hpt_queries


def evaluate(eval_loader, test=False):
    # Turn on evaluation mode which disables dropout.
    name='test' if test else 'val'
    model.eval()
    transform = get_transform()

    with torch.no_grad():
        for i_sample in range(1, opt.num_samples+1):
            did_plot = [False]*num_classes
            for batch_idx, (eval_info, eval_context, eval_target) in enumerate(eval_loader):
                # init batch
                eval_context = batch_to_device(eval_context, device)
                eval_target  = batch_to_device(eval_target, device)
                eval_all = merge_two_batch(eval_context, eval_target)
                num_episodes = len(eval_context)
                batch_size, mod_batch_sizes = get_batch_size(eval_target)

                # get img_queries
                img_queries = torch.from_numpy(np.array(eval_info[0]['add_cameras'])).float()

                # get true_images and hand_images
                true_images = load_images(eval_info[0]['add_images'], transform)
                _true_images = get_grid_image(true_images, 16, 3, 64, 64, nrow=4, pad_value=0)
                hand_images = load_images(eval_info[0]['hand_images'], transform)
                _hand_images = get_grid_image(hand_images, 15, 3, 64, 64, nrow=15, pad_value=0)
                _fst_image = get_grid_image(eval_all[0][0][:1], 1, 3, 64, 64, nrow=1, pad_value=0)
                _data_image = get_grid_image(eval_all[0][0], 15, 3, 64, 64, nrow=15, pad_value=0)

                ''' temporary '''
                assert len(eval_context) == 1
                assert len(eval_target) == 1
                cls = eval_info[0]['class']

                if (batch_idx+1) % 1 == 0:
                    print(batch_idx+1, '/', len(eval_loader))

                ''' per class '''
                # visualize per class
                if not did_plot[cls]:
                    # change flag
                    did_plot[cls] = True

                    # draw true_images and hand_images
                    writer.add_image('{}/gt-img-cls{}-i{}'.format(name, cls, i_sample), _true_images, 0)
                    writer.add_image('{}/hand-img-cls{}-i{}'.format(name, cls, i_sample), _hand_images, 0)
                    writer.add_image('{}/fst-img-cls{}-i{}'.format(name, cls, i_sample), _fst_image, 0)
                    writer.add_image('{}/data-img-cls{}-i{}'.format(name, cls, i_sample), _data_image, 0)

                    # init queries
                    mod_queries, num_mod_queries = [], []
                    for idx, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
                        # get queries
                        if mtype == 'image':
                            # image queries
                            _mod_queries, _num_mod_queries = get_visualization_queries_with_predefined_dist(num_episodes, device, img_queries)
                        elif mtype == 'haptic':
                            # haptic queries
                            _mod_queries, _num_mod_queries = get_visualization_queries_for_haptic(num_episodes, device)

                        # append to list
                        mod_queries += [_mod_queries]
                        num_mod_queries += [_num_mod_queries]

                    def draw_img_gen(num_context=0, use_img=True, use_hpt=True, use_first_img=True):
                        # use_first_img
                        if use_first_img:
                            first_image = [tuple([
                                eval_all[i][j][:1] if j//2 == 0 else None
                                for j in range(len(eval_all[i]))
                                ]) for i in range(num_episodes)]
                            new_eval_all = [tuple([
                                eval_all[i][j][1:] if j//2 == 0 else eval_all[i][j]
                                for j in range(len(eval_all[i]))
                                ]) for i in range(num_episodes)]
                        else:
                            new_eval_all = eval_all

                        # get context
                        new_eval_context, new_eval_target = trim_context_target(new_eval_all, num_context=num_context, use_img=use_img, use_hpt=use_hpt)
                        if new_eval_target[0][0] is None and new_eval_target[0][2] is None:
                            return

                        if use_first_img:
                            new_eval_context = merge_two_batch(new_eval_context, first_image)

                        # forward
                        outputs, _, _, _ = model(new_eval_context, new_eval_target, is_grayscale=opt.grayscale)

                        # generate
                        gens, _ = model.generate(new_eval_context, tuple(mod_queries), is_grayscale=opt.grayscale)

                        # visualize
                        img_gens = []
                        for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                            # get output and gen
                            output = outputs[idx]
                            gen = gens[idx]
                            _num_mod_queries = num_mod_queries[idx]

                            # visualize
                            if mtype == 'image':
                                # grayscale
                                if opt.grayscale:
                                    output = output.expand(output.size(0), nchannels, nheight, nwidth)
                                    gen = gen.expand(gen.size(0), nchannels, nheight, nwidth)

                                _gen = get_grid_image(gen, 16, 3, 64, 64, nrow=4, pad_value=0)
                                writer.add_image('{}/m{}-gen-cls{}-uimg{}-uhpt{}-ufimg{}-i{}-nc{}/img'.format(name, idx, cls, int(use_img), int(use_hpt), int(use_first_img), i_sample, num_context), _gen, 0)

                                # visualize predictions (image)
                                xs = get_visualization_image_data(idx, nchannels, nheight, nwidth, device, new_eval_context, new_eval_target, output, gen, _num_mod_queries, dataset_info['nviews'], nrow=4)
                                for i, x in enumerate(xs):
                                    writer.add_image('{}/m{}-cond-target-recon-gen-cls{}-uimg{}-uhpt{}-ufimg{}-i{}-nc{}/img'.format(name, idx, cls, int(use_img), int(use_hpt), int(use_first_img), i_sample, num_context), x, 0)
                                img_gens += [gen]
                                num_img_queries = _num_mod_queries
                            elif mtype == 'haptic':
                                # visualize predictions (haptic)
                                xs = get_visualization_haptic_data(idx, nchannels, nheight, device, new_eval_context, new_eval_target, output, gen, _num_mod_queries) 
                                for i, x in enumerate(xs):
                                    writer.add_image('{}/m{}-cond-target-recon-gen-cls{}-uimg{}-uhpt{}-ufimg{}-i{}-nc{}/hpt'.format(name, idx, cls, int(use_img), int(use_hpt), int(use_first_img), i_sample, num_context), x, 0)
                            else:
                                raise NotImplementedError

                    # run vis
                    use_imgs = [True, False] if eval_all[0][0] is not None else [False]
                    use_hpts = [True, False] if eval_all[0][2] is not None else [False]
                    for use_first_img in [False, True]:
                        for use_img in use_imgs:
                            for use_hpt in use_hpts:
                                for num_context in [0, 1, 2, 3, 4, 5, 10, 15]:
                                    draw_img_gen(num_context, use_img, use_hpt, use_first_img)

    return

# Run on val data
evaluate(val_loader, test=False)
logging('=' * 89, log_=False)
logging('| End of visualization ', log_=False)
logging('=' * 89, log_=False)
