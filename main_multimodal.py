# import
import os
import argparse
import time
import math
import datetime
import glob

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import datasets as dset
import models as net

from utils import get_time, logging, get_lrs, save_checkpoint, load_checkpoint
from utils import get_grid_image, get_image_from_values, get_plot
from utils import batch_to_device, merge_two_batch
from utils import sample_queries, sample_hand_queries, sample_random_queries, sample_random_hand_queries
from utils import get_visualization_image_data, get_visualization_haptic_data, get_combined_visualization_image_data


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

parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate')
parser.add_argument('--clip', type=float, default=0,  # 0.25,
                            help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                            help='upper epoch limit')
parser.add_argument('--start-epoch', type=int, default=1,
                            help='start epoch')
parser.add_argument('--start-batch-idx', type=int, default=0,
                            help='start batch-idx')

# training
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for test (default: 10)')
parser.add_argument('--optimizer', default='adam',
                    choices=['sgd', 'adam', 'adam-0.5'],
                    help='optimization methods: sgd | adam')
add_mod_parser = parser.add_mutually_exclusive_group(required=False)
add_mod_parser.add_argument('--add-mod',    dest='add_mod', action='store_true',  help='add module-specific losses')
add_mod_parser.add_argument('--no-add-mod', dest='add_mod', action='store_false', help='add module-specific losses')
parser.set_defaults(add_mod=False)
add_opposite_parser = parser.add_mutually_exclusive_group(required=False)
add_opposite_parser.add_argument('--add-opposite',    dest='add_opposite', action='store_true',  help='flag for adding batches in which target and context are swapped during training')
add_opposite_parser.add_argument('--no-add-opposite', dest='add_opposite', action='store_false', help='flag for adding batches in which target and context are swapped during training')
parser.set_defaults(add_opposite=False)

# annealing
parser.add_argument('--beta-init', type=float, default=1.0,
                    help='initial beta value for beta annealing')
parser.add_argument('--beta-fin',  type=float, default=1.0,
                    help='final beta value for beta annealing')
parser.add_argument('--beta-annealing', type=float, default=100000,
                    help='interval to annealing beta')
parser.add_argument('--std-init', type=float, default=math.sqrt(2.0),
                    help='initial std value for std annealing')
parser.add_argument('--std-fin', type=float, default=math.sqrt(0.5),
                    help='final std value for std annealing')
parser.add_argument('--std-annealing', type=float, default=None, #100000,
                    help='interval to annealing std')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, help='report interval')
parser.add_argument('--vis-interval', type=int, default=5000,
                    help='visualization interval')

# save
parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--cache', default=None, help='path to cache')
parser.add_argument('--experiment', default=None, help='name of experiment')
parser.add_argument('--exp-num', type=int, default=None,
                    help='experiment number')

# parse arguments
opt = parser.parse_args()

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if opt.cuda else "cpu")
opt.best_val1_loss = None

# generate cache folder
if opt.cache is None:
    opt.cache = 'experiments'
if opt.experiment is None:
    opt.experiment = '{}-{}-{}-addopp{}-exp{}'.format(
                      opt.dataset,
                      opt.model,
                      opt.optimizer,
                      1 if opt.add_opposite else 0,
                      opt.exp_num if opt.exp_num else 0,
                      )
opt.path = os.path.join(opt.cache, opt.experiment)
if opt.resume:
    listing = glob.glob(opt.path+'-2*')
    if len(listing) == 0:
        opt.path = '{}-{}'.format(opt.path, get_time())
    else:
        path_sorted = sorted(listing, key=lambda x: datetime.datetime.strptime(x, opt.path+'-%Y-%m-%d-%H:%M:%S.%f'))
        opt.path = path_sorted[-1]
        pass
else:
    opt.path = '{}-{}'.format(opt.path, get_time())
os.system('mkdir -p {}'.format(opt.path))

# print args
logging(str(opt), path=opt.path)


# init tensorboard
writer = SummaryWriter(opt.path)


# init dataset
train_loader, val1_loader, val2_loader, test_loader, dataset_info = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda)
num_modalities = dataset_info['num_modalities']
if val2_loader is not None:
    run_val2 = True
else:
    run_val2 = False


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

elif opt.model == 'cond-conv-apoe-multimodal-cgqn-v1':
    model = net.CondConvAPoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'cond-conv-apoe-multimodal-cgqn-v2':
    model = net.CondConvAPoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'cond-conv-apoe-multimodal-cgqn-v3':
    model = net.CondConvAPoEMultimodalCGQN_v3(dims=dataset_info['dims'], num_layers=1).to(device)
elif opt.model == 'cond-conv-apoe-multimodal-cgqn-v4':
    model = net.CondConvAPoEMultimodalCGQN_v4(dims=dataset_info['dims'], num_layers=1).to(device)

elif opt.model == 'cond-conv-apoe-multimodal-ml-cgqn-v1':
    model = net.CondConvAPoEMultimodalCGQN_v1(dims=dataset_info['dims'], num_layers=2).to(device)
elif opt.model == 'cond-conv-apoe-multimodal-ml-cgqn-v2':
    model = net.CondConvAPoEMultimodalCGQN_v2(dims=dataset_info['dims'], num_layers=2).to(device)

else:
    raise NotImplementedError('unknown model: {}'.format(opt.model))
logging(str(model), path=opt.path)


# init optimizer
if opt.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1./4.0, patience=0, verbose=True)
elif opt.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = None
elif opt.optimizer == 'adam-0.5':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = None
else:
    raise NotImplementedError('unknown optimizer: {}'.format(opt.optimizer))


# resume
load_checkpoint(model, optimizer, opt)


# msc
def get_batch_size(eval_target):
    mod_batch_sizes = []
    for i in range(num_modalities):
        mod_batch_sizes += [sum([target[i*2].size(0) for target in eval_target if target[i*2] is not None])]
    batch_size = sum(mod_batch_sizes)
    return batch_size, mod_batch_sizes

def get_visualization_queries_with_predefined_dist(num_episodes, device):
    # generate image with shuffled queries and random queries
    img_queries = sample_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False).to(device)
    num_img_queries = img_queries.size(0)
    img_queries = [img_queries for i in range(num_episodes)]
    return img_queries, num_img_queries

def get_visualization_queries_from_data(idx, eval_target, num_episodes, num_queries=32):
    # generate image with shuffled queries and random queries
    #hpt_queries = torch.cat([eval_target[i][idx*2+1] for i in range(num_episodes) if eval_target[i][idx*2+1] is not None], dim=0)
    hpt_queries = sample_hand_queries(nrow=4, ncol=8).to(device)
    num_hpt_queries = num_queries #32 #dataset_info['nviews']  # 15
    set_indices = [np.random.permutation(num_hpt_queries)[:num_hpt_queries] % hpt_queries.size(0) for i in range(num_episodes)]
    shuffled_hpt_queries = [torch.cat([hpt_queries[i:i+1] for i in indices], dim=0) for indices in set_indices]
    return shuffled_hpt_queries, num_hpt_queries


# define evaluate
def running_evaluate_for_val1(model):
    # set global variable
    global runinng_val1_data_iter

    # init
    model.eval()
    with torch.no_grad():
        # get one minibatch
        try:
            _, eval_context, eval_target = runinng_val1_data_iter.next()
        except:
            runinng_val1_data_iter = iter(val1_loader)
            _, eval_context, eval_target = runinng_val1_data_iter.next()

        # init batch
        eval_context = batch_to_device(eval_context, device)
        eval_target  = batch_to_device(eval_target, device)
        num_episodes = len(eval_context)
        batch_size, mod_batch_sizes = get_batch_size(eval_target)

        # forward
        _, _, loss, info = model(eval_context, eval_target)

        # unpack info
        loss_likelihood, loss_kl = info['likelihood'], info['kl']
        loss_mod_likelihoods = info['mod_likelihoods']

        # add to total_loss
        total_loss = loss.item() / batch_size * num_modalities

    return total_loss

if run_val2:
    def running_evaluate_for_val2(model):
        # set global variable
        global runinng_val2_data_iter

        # init
        model.eval()
        with torch.no_grad():
            # get one minibatch
            try:
                _, eval_context, eval_target = runinng_val2_data_iter.next()
            except:
                runinng_val2_data_iter = iter(val2_loader)
                _, eval_context, eval_target = runinng_val2_data_iter.next()

            # init batch
            eval_context = batch_to_device(eval_context, device)
            eval_target  = batch_to_device(eval_target, device)
            num_episodes = len(eval_context)
            batch_size, mod_batch_sizes = get_batch_size(eval_target)

            # forward
            _, _, loss, info = model(eval_context, eval_target)

            # unpack info
            loss_likelihood, loss_kl = info['likelihood'], info['kl']
            loss_mod_likelihoods = info['mod_likelihoods']

            # add to total_loss
            total_loss = loss.item() / batch_size * num_modalities

        return total_loss

def evaluate(eval_loader, test=False):
    # Turn on evaluation mode which disables dropout.
    name='test' if test else 'val'
    model.eval()
    total_loss = 0.
    total_batch_size = 0
    total_mod_likelihoods = [0]*num_modalities
    total_mod_batch_sizes = [0]*num_modalities
    latents = []
    with torch.no_grad():
        for batch_idx, (_, eval_context, eval_target) in enumerate(eval_loader):
            # init batch
            eval_context = batch_to_device(eval_context, device)
            eval_target  = batch_to_device(eval_target, device)
            num_episodes = len(eval_context)
            batch_size, mod_batch_sizes = get_batch_size(eval_target)

            # forward
            outputs, latent, loss, info = model(eval_context, eval_target)

            # unpack info
            loss_likelihood, loss_kl = info['likelihood'], info['kl']
            loss_mod_likelihoods = info['mod_likelihoods']

            # add to latents
            latents += [latent] if latent is not None else []

            # add to total_loss
            total_loss += loss.item() * num_modalities #/ batch_size * num_episodes
            total_batch_size += batch_size
            for i in range(num_modalities):
                #total_mod_likelihoods[i] += loss_mod_likelihoods[i].item() / mod_batch_sizes[i] * num_episodes if loss_mod_likelihoods[i] is not None else 0
                total_mod_likelihoods[i] += loss_mod_likelihoods[i].item() if loss_mod_likelihoods[i] is not None else 0
                total_mod_batch_sizes[i] += mod_batch_sizes[i]

            # visualize prediction
            if batch_idx + 1 == len(eval_loader):
                # init queries
                mod_queries, num_mod_queries = [], []
                for idx, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
                    # get queries
                    if mtype == 'image':
                        # image queries
                        _mod_queries, _num_mod_queries = get_visualization_queries_with_predefined_dist(num_episodes, device)
                    elif mtype == 'haptic':
                        # haptic queries
                        _mod_queries, _num_mod_queries = get_visualization_queries_from_data(idx, eval_target, num_episodes)

                    # append to list
                    mod_queries += [_mod_queries]
                    num_mod_queries += [_num_mod_queries]

                # generate
                gens, latent = model.generate(eval_context, tuple(mod_queries))

                # visualize
                img_gens = []
                for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                    # get output and gen
                    output = outputs[idx]
                    gen = gens[idx]
                    _num_mod_queries = num_mod_queries[idx]

                    # visualize
                    if mtype == 'image':
                        # visualize predictions (image)
                        xs = get_visualization_image_data(idx, nchannels, nheight, nwidth, device, eval_context, eval_target, output, gen, _num_mod_queries, dataset_info['nviews'])
                        for i, x in enumerate(xs):
                            writer.add_image('{}/m{}-cond-target-recon-gensh-genrd-b{}-i{}/img'.format(name, idx, batch_idx, i), x, epoch)
                        # temporary
                        img_gens += [gen]
                        num_img_queries = _num_mod_queries
                    elif mtype == 'haptic':
                        # visualize predictions (haptic)
                        xs = get_visualization_haptic_data(idx, nchannels, nheight, device, eval_context, eval_target, output, gen, _num_mod_queries) 
                        for i, x in enumerate(xs):
                            writer.add_image('{}/m{}-cond-target-recon-gensh-genrd-b{}-i{}/hpt'.format(name, idx, batch_idx, i), x, epoch)
                    else:
                        raise NotImplementedError

                # visualize combined image
                xs = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_gens, num_img_queries, min(4, len(eval_context)))
                for i, x in enumerate(xs):
                    writer.add_image('{}/cond-target-recon-gensh-genrd-b{}-i{}/img'.format(name, batch_idx, i), x, epoch)

    return total_loss / total_batch_size

# define train
def train(train_loader, model, optimizer, epoch, start_batch_idx=0):
    # init
    start_time = time.time()
    model.train()
    total_loss = 0.
    total_likelihood = 0.
    total_mod_likelihoods = [0.]*num_modalities
    total_kl = 0.
    total_batch_size = 0
    total_mod_batch_sizes = [0]*num_modalities
    for _batch_idx, (_, train_context, train_target) in enumerate(train_loader):
        # init batch_idx
        batch_idx = _batch_idx + start_batch_idx
        i_episode = (epoch-1)*len(train_loader) + batch_idx

        # init beta and std
        beta = opt.beta_init + (opt.beta_fin - opt.beta_init) / float(opt.beta_annealing) * float(min(opt.beta_annealing, i_episode))
        std  = opt.std_init  + (opt.std_fin  - opt.std_init)  / float(opt.std_annealing)  * float(min(opt.std_annealing, i_episode)) if opt.std_annealing is not None else None

        # init batch
        train_context = batch_to_device(train_context, device)
        train_target  = batch_to_device(train_target, device)

        # add additional datasets
        _train_context = []
        _train_target  = []
        if opt.add_opposite:
            _train_context += train_target
            _train_target  += train_context
        train_context += _train_context
        train_target  += _train_target

        # init numbers
        num_episodes = len(train_context)
        batch_size, mod_batch_sizes = get_batch_size(train_target)

        # init grad
        model.zero_grad()

        ''' ELBO '''
        # forward (joint observation)
        outputs, latent, loss, info = \
                model(train_context, train_target, beta=beta) if opt.std_annealing is None \
                else model(train_context, train_target, beta=beta, std=std)

        # backward (joint observation)
        loss.backward()

        # forward (module-specific observation)
        if opt.add_mod:
            for m in range(num_modalities):
                # check target is not empty
                is_not_empty = True in [train_target[i][m*2] is not None for i in range(num_episodes)]
                if is_not_empty:
                    # fetch module-specific data
                    mod_train_context = []
                    mod_train_target  = []
                    for i in range(num_episodes):
                        if train_target[i][m*2] is not None:
                            _mod_train_context = [None, None]*num_modalities
                            _mod_train_context[m*2]   = train_context[i][m*2]
                            _mod_train_context[m*2+1] = train_context[i][m*2+1]
                            _mod_train_context = tuple(_mod_train_context)
                            mod_train_context += [_mod_train_context]

                            _mod_train_target = [None, None]*num_modalities
                            _mod_train_target[m*2]    = train_target[i][m*2]
                            _mod_train_target[m*2+1]  = train_target[i][m*2+1]
                            _mod_train_target = tuple(_mod_train_target)
                            mod_train_target += [_mod_train_target]

                    # forward (module-specific observation)
                    _, _, mod_loss, _ = \
                            model(mod_train_context, mod_train_target, beta=beta) if opt.std_annealing is None \
                            else model(mod_train_context, mod_train_target, beta=beta, std=std)

                    # backward (module-specific observation)
                    if mod_loss is not None:
                        mod_loss.backward()


        # unpack info
        loss_likelihood, loss_kl = info['likelihood'], info['kl']
        loss_mod_likelihoods = info['mod_likelihoods']

        # `clip_grad_norm` helps prevent the exploding gradient problem in continuous data with gaussian likelihood
        if opt.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        # update
        optimizer.step()

        # add to total loss
        cur_loss = loss.item()
        cur_likelihood = loss_likelihood.item()
        cur_kl = loss_kl.item()
        cur_mod_likelihoods = [loss_mod_likelihood.item() if loss_mod_likelihood is not None else None for loss_mod_likelihood in loss_mod_likelihoods]
        total_loss += cur_loss * num_modalities #/ batch_size * num_episodes
        total_likelihood += cur_likelihood * num_modalities #/ batch_size * num_episodes
        total_kl += cur_kl * num_modalities #/ batch_size * num_episodes
        total_batch_size += batch_size
        for i in range(num_modalities):
            #total_mod_likelihoods[i] += cur_mod_likelihoods[i] / mod_batch_sizes[i] * num_episodes if cur_mod_likelihoods[i] is not None else 0
            total_mod_likelihoods[i] += cur_mod_likelihoods[i] if cur_mod_likelihoods[i] is not None else 0
            total_mod_batch_sizes[i] += mod_batch_sizes[i]

        # print
        if (batch_idx+1) % opt.log_interval == 0:
            # plot running val
            val1_loss = running_evaluate_for_val1(model)
            val2_loss = running_evaluate_for_val2(model) if run_val2 else -1.
            model.train()

            # set log info
            elapsed = time.time() - start_time
            lr_min, lr_max = get_lrs(optimizer)

            # print
            logging('| epoch {:3d} | {:5d}/{:5d} '
                    '| lr_min {:02.4f} | lr_max {:02.4f} | ms/step {:5.2f} '
                    '| beta {:02.4f} '
                    '| loss {:5.8f} | lk+kl {:5.8f} | likelihood {:5.8f} | kl {:5.8f} '
                    '{}'
                    '| val1 loss (lk+kl) {:5.8f} '
                    '| val2 loss (lk+kl) {:5.8f} '
                    .format(
                    epoch,
                    batch_idx+1, len(train_loader),
                    lr_min, lr_max, elapsed * 1000 / opt.log_interval,
                    beta,
                    cur_loss / batch_size * num_modalities,
                    (cur_likelihood + cur_kl) / batch_size * num_modalities,
                    cur_likelihood / batch_size * num_modalities,
                    cur_kl / batch_size * num_modalities,
                    ''.join(['| m{}_{}_lk {} '.format(
                        i,
                        mtype,
                        '{:5.8f}'.format(cur_mod_likelihoods[i] / mod_batch_sizes[i]) if cur_mod_likelihoods[i] is not None else '-.--------',
                        ) for i, (_, _, _, _, mtype) in enumerate(dataset_info['dims'])]),
                    val1_loss,
                    val2_loss,
                    ),
                    path=opt.path)

            # write to tensorboard
            writer.add_scalar('train/loss/step', cur_loss / batch_size * num_modalities, i_episode)
            writer.add_scalar('train/lk+kl/step', (cur_likelihood + cur_kl) / batch_size * num_modalities, i_episode)
            writer.add_scalar('train/likelihood/step', cur_likelihood / batch_size * num_modalities, i_episode)
            for i, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
                if cur_mod_likelihoods[i] is not None:
                    writer.add_scalar('train/m{}_{}_lk/step'.format(i, 'img' if mtype == 'image' else 'hpt'), cur_mod_likelihoods[i] / mod_batch_sizes[i], i_episode)
            writer.add_scalar('train/kl/step', cur_kl / batch_size * num_modalities, i_episode)
            writer.add_scalar('train/beta', beta, i_episode)
            writer.add_scalar('val1/loss/step',  val1_loss, i_episode)
            writer.add_scalar('val1/lk+kl/step', val1_loss, i_episode)
            writer.add_scalar('val2/loss/step',  val2_loss, i_episode)
            writer.add_scalar('val2/lk+kl/step', val2_loss, i_episode)
            if std is not None:
                writer.add_scalar('train/std', std, i_episode)

            # reset log info
            start_time = time.time()

        if batch_idx+1 == len(train_loader):
            # print
            logging('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| loss {:5.8f} | lk+kl {:5.8f} | likelihood {:5.8f} | kl {:5.8f} '
                    '{}'
                    .format(
                    epoch,
                    batch_idx+1, len(train_loader),
                    total_loss / total_batch_size, #len(train_loader.dataset),
                    (total_likelihood+total_kl) / total_batch_size, #/ len(train_loader.dataset),
                    total_likelihood / total_batch_size, #len(train_loader.dataset),
                    total_kl / total_batch_size, #len(train_loader.dataset),
                    ''.join(['| m{}_{}_lk {:5.8f} '.format(i, mtype, total_mod_likelihoods[i] / total_mod_batch_sizes[i]) for i, (_, _, _, _, mtype) in enumerate(dataset_info['dims']) if total_mod_batch_sizes[i] > 0])
                    ),
                    path=opt.path)

            # write to tensorboard
            writer.add_scalar('train/loss', total_loss / total_batch_size, epoch) #len(train_loader.dataset), epoch)
            writer.add_scalar('train/likelihood', total_likelihood / total_batch_size, epoch) #len(train_loader.dataset), epoch)
            for i, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
                if total_mod_batch_sizes[i] > 0:
                    writer.add_scalar('train/m{}_{}_lk/step'.format(i, 'img' if mtype == 'image' else 'hpt'), total_mod_likelihoods[i] / total_mod_batch_sizes[i], epoch) #len(train_loader.dataset), epoch)
            writer.add_scalar('train/kl', total_kl / total_batch_size, epoch) #len(train_loader.dataset), epoch)
            writer.add_scalar('train/lk+kl', (total_likelihood + total_kl) / total_batch_size, epoch) #len(train_loader.dataset), epoch)

        if (batch_idx+1) % opt.vis_interval == 0 or (batch_idx+1 == len(train_loader)):
            # generate image with shuffled queries and random queries
            model.eval()
            with torch.no_grad():
                # init queries
                mod_queries, num_mod_queries = [], []
                for idx, (_, _, _, _, mtype) in enumerate(dataset_info['dims']):
                    # get queries
                    if mtype == 'image':
                        # image queries
                        _mod_queries, _num_mod_queries = get_visualization_queries_with_predefined_dist(num_episodes, device)
                    elif mtype == 'haptic':
                        # haptic queries
                        _mod_queries, _num_mod_queries = get_visualization_queries_from_data(idx, train_target, num_episodes)

                    # append to list
                    mod_queries += [_mod_queries]
                    num_mod_queries += [_num_mod_queries]

                # generate
                gens, latent = model.generate(train_context, tuple(mod_queries))
            model.train()

            # visualize
            img_gens = []
            for idx, (nchannels, nheight, nwidth, _, mtype) in enumerate(dataset_info['dims']):
                # get output and gen
                output = outputs[idx]
                gen = gens[idx]
                _num_mod_queries = num_mod_queries[idx]

                # visualize
                if mtype == 'image':
                    # visualize predictions (image)
                    xs = get_visualization_image_data(idx, nchannels, nheight, nwidth, device, train_context, train_target, output, gen, _num_mod_queries, dataset_info['nviews'])
                    for i, x in enumerate(xs):
                        writer.add_image(
                                'train/m{}-cond-target-recon-gensh-genrd-i{}/img'.format(idx, i),
                                x, i_episode)
                    # temporary
                    img_gens += [gen]
                    num_img_queries = _num_mod_queries
                elif mtype == 'haptic':
                    # visualize predictions (haptic)
                    xs = get_visualization_haptic_data(idx, nchannels, nheight, device, train_context, train_target, output, gen, _num_mod_queries) 
                    for i, x in enumerate(xs):
                        writer.add_image(
                                'train/m{}-cond-target-recon-gensh-genrd-i{}/hpt'.format(idx, i),
                                x, i_episode)
                else:
                    raise NotImplementedError

            # visualize combined image
            xs = get_combined_visualization_image_data(opt.dataset, dataset_info['dims'], img_gens, num_img_queries, min(4, len(train_context)))
            for i, x in enumerate(xs):
                writer.add_image('train/cond-target-recon-gensh-genrd-i{}/img'.format(i), x, i_episode)

            # save model
            with open(os.path.join(opt.path, 'model.pt'), 'wb') as f:
                torch.save(model, f)
            save_checkpoint({
                'epoch': epoch+1 if (batch_idx+1) == len(train_loader) else epoch,
                'batch_idx': (batch_idx+1) % len(train_loader),
                'model': opt.model,
                'state_dict': model.state_dict(),
                'best_val1_loss': best_val1_loss,
                'optimizer' : optimizer.state_dict(),
            }, opt, False)

            # flush writer
            writer.flush()

        if batch_idx+1 == len(train_loader):
            writer.flush()
            break

# init
best_val1_loss = opt.best_val1_loss

# at any point you can hit ctrl + c to break out of training early.
try:
    for epoch in range(opt.start_epoch, opt.epochs+1):
        epoch_start_time = time.time()

        # train
        train(train_loader, model, optimizer, epoch, opt.start_batch_idx)
        opt.start_batch_idx = 0

        # eval valid
        val1_loss = evaluate(val1_loader)
        val2_loss = evaluate(val2_loader) if run_val2 else -1.

        # update lr scheduler
        if scheduler is not None:
            scheduler.step(val1_loss)

        # logging
        logging('-' * 89, path=opt.path)
        logging('| end of epoch {:3d} | time: {:5.2f}s '
                '| valid loss (lk+kl) {:5.8f} '.format(
                epoch, (time.time() - epoch_start_time),
                val1_loss),
                path=opt.path)
        logging('-' * 89, path=opt.path)

        # write to tensorboard
        writer.add_scalar('val1/loss',  val1_loss, epoch)
        writer.add_scalar('val1/lk+kl', val1_loss, epoch)
        writer.add_scalar('val2/loss',  val2_loss, epoch)
        writer.add_scalar('val2/lk+kl', val2_loss, epoch)

        # Save the model for each epoch
        with open(os.path.join(opt.path, 'states-e{}.pt'.format(epoch)), 'wb') as f:
            torch.save(model.state_dict(), f)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val1_loss or val1_loss < best_val1_loss:
            with open(os.path.join(opt.path, 'best-model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val1_loss = val1_loss
        else:
            pass

        # flush writer
        writer.flush()

except KeyboardInterrupt:
    writer.flush()
    logging('-' * 89, path=opt.path)
    logging('Exiting from training early', path=opt.path)
