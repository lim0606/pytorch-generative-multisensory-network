import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.convdraw3 import ConvLSTMCell, StackedConvDrawEncoderCell, StackedConvDrawDecoderCell, ConvLSTM
from models.reparam import NormalDistributionConv2d, NormalDistributionConvTranspose2d
from models.image_representation import ContextNetwork as ImageContextNetwork
from models.image_representation import CameraEncoder
from models.image_renderer import ImageRenderer
from models.haptic_representation import ContextNetwork as HapticContextNetwork
from models.haptic_representation import HandEncoder
from models.haptic_renderer import HapticRenderer

from utils import loss_kld_gaussian_vs_gaussian, loss_recon_gaussian_w_fixed_var
from utils import logprob_gaussian, logprob_gaussian_w_fixed_var
from utils import broadcast_representation, sum_tensor_per_episode
from utils import pack_tensor_list, flatten_packed_tensor
from utils import merge_two_batch, pad_sequence, get_reversed_tensor, get_reversed_sequence, sort_padded_sequence
from utils import rgb2gray


def combine_reps(inputs):
    '''
    Input:
        inputs: a list of tuples, each of which (reps, context_sizes)
                ex) [(reps_1, context_sizes_1), (reps_2, context_sizes_2), ... ]
    Output:
        reps: concatenated reps
        context_sizes: concatenated context_sizes
    '''
    # check sizes
    assert len(inputs) > 0
    _reps, _context_sizes = inputs[0]
    for _rs, _css in inputs:
        assert len(_reps) == len(_rs)
        assert len(_context_sizes) == len(_css)

    # preprocess
    items = []
    for item in inputs:
        items += list(item)

    # combine
    reps, context_sizes = [], []
    for rep_context_size in zip(*items):
        rep = torch.cat(rep_context_size[::2], dim=0)
        reps += [rep]
        context_size = sum(rep_context_size[1::2])
        context_sizes += [context_size]
    return reps, context_sizes


class ProductOfExperts(nn.Module):
    ''' copied from https://github.com/mhw32/multimodal-vae-public/blob/master/mnist/model.py '''
    ''' Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    ''' 
    def forward(self, mu, logvar, eps=1e-8, dim=0):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=dim) / torch.sum(T, dim=dim)
        pd_var    = 1. / torch.sum(T, dim=dim)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class CGQNConvDraw(nn.Module):
    def __init__(self,
                 dims, #=[(3, 64, 64, 7, 'image'), (132, 1, 1, 7, 'haptic')],
                 #im_height,  # image height
                 #im_channels,  # number of channels in image
                 #nc_img_query,  # kernel size (number of channels) for query
                 #hp_height,  # haptic height
                 #hp_channels,  # number of channels in haptic
                 #nc_hpt_query,  # kernel size (number of channels) for query
                 nc_enc,  # kernel size (number of channels) for encoder
                 nc_lstm,  # kernel size (number of channels) for lstm
                 nc_context,  # kernel size (number of channels) for representation
                 nz,  # size of latent variable
                 num_steps,  # number of steps in Draw
                 num_layers,  # number of StackedConvDrawEncoderCell/StackedConvDrawDecoderCell layers
                 ):
        super().__init__()
        # check conditions
        assert len(dims) > 0
        for dim in dims:
            assert len(dim) == 5, dim 
            channels, height, width, nc_query, mtype = dim
            #assert height == width
            assert mtype in ['image', 'haptic']

        # find im_height
        im_heights = [height for _, height, _, _, mtype in dims if mtype == 'image']
        if len(im_heights) == 0:
            im_height = 64
        else:
            im_height = im_heights[0]
            for _im_height in im_heights[1:]:
                assert im_height == _im_height
        im_widths = [width for _, _, width, _, mtype in dims if mtype == 'image']
        if len(im_widths) == 0:
            im_width = 64
        else:
            im_width = im_widths[0]
            for _im_width in im_widths[1:]:
                assert im_width == _im_width

        # init
        self.dims = dims
        #self.im_height = im_height
        #self.im_channels = im_channels
        #self.nc_img_query = nc_img_query
        #self.hp_height = hp_height
        #self.hp_channels = hp_channels
        #self.nc_hpt_query = nc_hpt_query
        self.nc_enc = nc_enc
        self.nc_lstm = nc_lstm
        self.nc_context = nc_context
        self.nz = nz
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.z_height = im_height // 4  # height of feature map size for z
        self.z_width = im_width // 4  # height of feature map size for z

        # init
        self.num_multimodalities = len(dims)

        # define networks
        self.rnn_q = StackedConvDrawEncoderCell(
                nc_context*2 + self.num_multimodalities,
                #nc_context,
                nc_lstm,
                nz=nz,
                kernel_size=5,
                padding=2,
                num_layers=num_layers,
                )

        self.rnn_p = StackedConvDrawDecoderCell(
                nc_context + self.num_multimodalities,
                nc_lstm,
                nz=nz,
                kernel_size=5,
                padding=2,
                num_layers=num_layers,
                )

        renderers, query_encoders = [], []
        for dim in dims:
            channels, height, width, nc_query, mtype = dim
            if mtype == 'image':
                renderers += [ImageRenderer(
                        im_height = height,
                        im_width = width,
                        im_channels = channels,
                        nc_query = nc_query,
                        nc_enc = self.nc_enc,
                        nc_lstm = self.nc_lstm,
                        nz_per_step = self.num_layers * self.nz,
                        z_num_steps = self.num_steps,
                        z_height = self.z_height,
                        z_width = self.z_width,
                        num_steps = self.num_steps,
                        )]
                query_encoders += [CameraEncoder()]
            elif mtype == 'haptic':
                renderers += [HapticRenderer(
                        hp_height = height,
                        hp_width = width,
                        hp_channels = channels,
                        nc_query = nc_query,
                        nc_enc = self.nc_enc,
                        nc_lstm = self.nc_lstm,
                        nz_per_step = self.num_layers * self.nz,
                        z_num_steps = self.num_steps,
                        z_height = self.z_height,
                        z_width = self.z_width,
                        num_steps = self.num_steps,
                        )]
                query_encoders += [HandEncoder()]
        self.renderers = nn.ModuleList(renderers)
        self.query_encoders = nn.ModuleList(query_encoders)
        self.experts = ProductOfExperts()
        self.modality_identifier = torch.eye(self.num_multimodalities).unsqueeze(2).unsqueeze(2) # #modal x #modal x 1 x 1

    def forward(self,
                reps_context_sizes_pairs_c,
                #reps_context, context_sizes,
                reps_context_sizes_pairs_t,
                #reps_target, target_sizes,
                input_tuples,
                #img_target=None, img_queries=None, img_batch_sizes=[], img_target_indices=[],
                #hpt_target=None, hpt_queries=None, hpt_batch_sizes=[], hpt_target_indices=[],
                num_steps=None, beta=1.0, std=1.0,
                is_grayscale=False,
                do_sum=True):
        # init
        num_episodes = len(reps_context_sizes_pairs_c[0][0])
        #assert len(set([index for _, _, mod_target_indices, _ in input_tuples for index in mod_target_indices])) == num_episodes
        loss_kl = 0
        weight = next(self.parameters())
        if self.modality_identifier.device != weight.device:
            self.modality_identifier = self.modality_identifier.to(weight.device)

        ''' forward posterior / prior '''
        # init states
        states_p  = self.rnn_p.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        states_q  = self.rnn_q.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        hiddens_p = [state_p[0] for state_p in states_p]
        #hiddens_q = [state_q[0] for state_q in states_q]
        latents = []
        inputs_q = []
        inputs_p = []
        init_input_q = False
        init_input_p = False
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # aggregate observations (posterior)
            if not init_input_q:
                for m in range(self.num_multimodalities):
                    # mod identifier
                    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                    # unpack input and combined contexts
                    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                    reps_context = pad_sequence(reps_context, context_sizes)
                    reps_context = torch.sum(reps_context, dim=1)
                    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                    inputs_p += [input_p.unsqueeze(2)]

                    reps_target, target_sizes = reps_context_sizes_pairs_t[m]
                    reps_target = pad_sequence(reps_target, target_sizes)
                    reps_target = torch.sum(reps_target, dim=1)
                    reps_target = reps_target.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_q = torch.cat([reps_target, reps_context, mod_identifier], dim=1)
                    inputs_q += [input_q.unsqueeze(2)]

                # concat inputs_q
                inputs_p = torch.cat(inputs_p, dim=2)

                # concat inputs_q
                inputs_q = torch.cat(inputs_q, dim=2)

                # update flag
                init_input_q = True
                init_input_p = True

            # forward posterior
            means_q, logvars_q, hiddens_q, states_q = self.rnn_q(inputs_q, states_q, hiddens_p)

            # get experts
            # means_q : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_q, _logvars_q = self.experts(means_q[j], logvars_q[j], dim=2)
                means_q[j]   = _means_q
                logvars_q[j] = _logvars_q

            # sample z from posterior
            zs = self.rnn_q.sample(means_q, logvars_q)

            # aggregate observations (prior)
            if not init_input_p:
                #for m in range(self.num_multimodalities):
                #    # mod identifier
                #    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                #    # unpack input and combined contexts
                #    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                #    reps_context = pad_sequence(reps_context, context_sizes)
                #    reps_context = torch.sum(reps_context, dim=1)
                #    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                #    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                #    inputs_p += [input_p.unsqueeze(2)]

                ## concat inputs_q
                #inputs_p = torch.cat(inputs_p, dim=2)

                ## update flag
                #init_input_p = True
                raise ValueError

            # unsqueeze zs and expand it
            _zs = [z.unsqueeze(2).expand(num_episodes, self.nz, self.num_multimodalities, self.z_height, self.z_width)  for z in zs]

            # forward prior
            _, means_p, logvars_p, hiddens_p, states_p = self.rnn_p(inputs_p, states_p, latents_q=_zs)

            # get experts
            # means_p : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_p, _logvars_p = self.experts(means_p[j], logvars_p[j], dim=2)
                means_p[j]   = _means_p
                logvars_p[j] = _logvars_p

            # append z to latent
            latents += [torch.cat(zs, dim=1).unsqueeze(1)] if len(zs) > 1 else [zs[0].unsqueeze(1)]

            # update accumulated KL
            for j in range(self.num_layers):
                loss_kl += loss_kld_gaussian_vs_gaussian(means_q[j], logvars_q[j], means_p[j], logvars_p[j], do_sum=do_sum)

        ''' likelihood '''
        info = {}
        info['mod_likelihoods'] = []
        loss_likelihood = 0 if do_sum else loss_kl.new_zeros(loss_kl.size())
        mean_recons = []
        for idx, (dim, input_tuple) in enumerate(zip(self.dims, input_tuples)):
            channels, height, width, _, mtype = dim
            mod_target, mod_queries, mod_target_indices, mod_batch_sizes = input_tuple
            if len(mod_queries) > 0:# is not None:
                num_mod_data = len(mod_target)
                assert sum(mod_batch_sizes) == num_mod_data

                # run renderer (likelihood)
                mod_mean_recon = self._forward_renderer(idx, mod_queries, latents, num_episodes, mod_batch_sizes, mod_target_indices)

                # convert to gray scale
                if mtype == 'image' and is_grayscale:
                    mod_mean_recon = rgb2gray(mod_mean_recon)
                    mod_target = rgb2gray(mod_target)

                # estimate recon loss
                loss_mod_likelihood = loss_recon_gaussian_w_fixed_var(mod_mean_recon, mod_target, std=std, add_logvar=False, do_sum=do_sum)

                # estimate recon loss without std
                loss_mod_likelihood_nostd = loss_recon_gaussian_w_fixed_var(mod_mean_recon.detach(), mod_target, do_sum=do_sum)
            else:
                mod_mean_recon = reps_context.new_zeros(0, channels, height, width)
                loss_mod_likelihood = None
                loss_mod_likelihood_nostd = None

            # append to list
            mean_recons += [mod_mean_recon]
            info['mod_likelihoods'] += [loss_mod_likelihood_nostd]

            # add to loss_likelihood
            if loss_mod_likelihood is not None:
                # sum to each episode
                if not do_sum:
                    _mod_batch_sizes = [0] + np.cumsum(mod_batch_sizes).tolist()
                    for i, t_idx in enumerate(mod_target_indices):
                        loss_likelihood[t_idx] += torch.sum(loss_mod_likelihood[_mod_batch_sizes[i]:_mod_batch_sizes[i+1]])
                else:
                    loss_likelihood += loss_mod_likelihood

        ''' loss '''
        # sum loss
        loss = loss_likelihood + beta * loss_kl

        # additional loss info
        info['likelihood'] = loss_likelihood.detach()
        info['kl'] = loss_kl.detach()

        ## temporary
        #img_mean_recon, hpt_mean_recon = mean_recons[0], mean_recons[1]
        #info['img_likelihood'] = info['mod_likelihoods'][0]
        #info['hpt_likelihood'] = info['mod_likelihoods'][1]

        # return
        #return img_mean_recon, hpt_mean_recon, None, loss, info
        return mean_recons, latents, loss, info

    def generate(self,
                 reps_context_sizes_pairs_c,
                 #reps_context, context_sizes,
                 input_tuples,
                 #img_queries, img_batch_sizes,
                 #hpt_queries, hpt_batch_sizes,
                 num_steps=None,
                 is_grayscale=False,
                 ):
        # init
        num_episodes = len(reps_context_sizes_pairs_c[0][0])
        weight = next(self.parameters())
        if self.modality_identifier.device != weight.device:
            self.modality_identifier = self.modality_identifier.to(weight.device)

        ''' forward posterior / prior '''
        # init states
        states_p  = self.rnn_p.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        #hiddens_p = [state_p[0] for state_p in states_p]
        latents = []
        inputs_p = []
        init_input_p = False
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # forward prior (prob)
            means_p, logvars_p = self.rnn_p.forward_prob(states_p)

            # get experts
            # means_p : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_p, _logvars_p = self.experts(means_p[j], logvars_p[j], dim=2)
                means_p[j]   = _means_p
                logvars_p[j] = _logvars_p

            # sample z from prior
            zs = self.rnn_p.sample(means_p, logvars_p)

            # aggregate observations (prior)
            if not init_input_p:
                for m in range(self.num_multimodalities):
                    # mod identifier
                    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                    # unpack input and combined contexts
                    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                    reps_context = pad_sequence(reps_context, context_sizes)
                    reps_context = torch.sum(reps_context, dim=1)
                    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                    inputs_p += [input_p.unsqueeze(2)]

                # concat inputs_q
                inputs_p = torch.cat(inputs_p, dim=2)

                # update flag
                init_input_p = True

            # unsqueeze zs and expand it
            _zs = [z.unsqueeze(2).expand(num_episodes, self.nz, self.num_multimodalities, self.z_height, self.z_width)  for z in zs]

            # forward prior (rnn)
            hiddens_p, states_p = self.rnn_p.forward_rnn(inputs_p, states_p, _zs)

            # append z to latent
            latents += [torch.cat(zs, dim=1).unsqueeze(1)] if len(zs) > 1 else [zs[0].unsqueeze(1)]

        ''' forward renderers '''
        mean_recons = []
        for idx, (dim, input_tuple) in enumerate(zip(self.dims, input_tuples)):
            channels, height, width, _, mtype = dim
            mod_queries, mod_batch_sizes = input_tuple

            # forward image renderer
            if len(mod_queries) > 0:
                # forward image renderer
                mod_mean_recon = self._forward_renderer(idx, mod_queries, latents, num_episodes, mod_batch_sizes)

                # convert to gray scale
                if mtype == 'image' and is_grayscale:
                    mod_mean_recon = rgb2gray(mod_mean_recon)
            else:
                mod_mean_recon = reps_context.new_zeros(0, channels, height, width)

            # append to list
            mean_recons += [mod_mean_recon]

        ## temporary
        #img_mean_recon, hpt_mean_recon = mean_recons[0], mean_recons[1]

        # return
        #return img_mean_recon, hpt_mean_recon, None
        return mean_recons, None

    def predict(self,
                reps_context_sizes_pairs_c,
                reps_context_sizes_pairs_t,
                input_tuples,
                num_steps=None, beta=1.0, std=1.0,
                is_grayscale=False,
                use_uint8=True):
        # init
        num_episodes = len(reps_context_sizes_pairs_c[0][0])
        logprob_kl = 0
        loss_kl = 0
        weight = next(self.parameters())
        if self.modality_identifier.device != weight.device:
            self.modality_identifier = self.modality_identifier.to(weight.device)

        ''' forward posterior / prior '''
        # init states
        states_p  = self.rnn_p.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        states_q  = self.rnn_q.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        hiddens_p = [state_p[0] for state_p in states_p]
        #hiddens_q = [state_q[0] for state_q in states_q]
        latents = []
        inputs_q = []
        inputs_p = []
        init_input_q = False
        init_input_p = False
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # aggregate observations (posterior)
            if not init_input_q:
                for m in range(self.num_multimodalities):
                    # mod identifier
                    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                    # unpack input and combined contexts
                    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                    reps_context = pad_sequence(reps_context, context_sizes)
                    reps_context = torch.sum(reps_context, dim=1)
                    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                    inputs_p += [input_p.unsqueeze(2)]

                    reps_target, target_sizes = reps_context_sizes_pairs_t[m]
                    reps_target = pad_sequence(reps_target, target_sizes)
                    reps_target = torch.sum(reps_target, dim=1)
                    reps_target = reps_target.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_q = torch.cat([reps_target, reps_context, mod_identifier], dim=1)
                    inputs_q += [input_q.unsqueeze(2)]

                # concat inputs_q
                inputs_p = torch.cat(inputs_p, dim=2)

                # concat inputs_q
                inputs_q = torch.cat(inputs_q, dim=2)

                # update flag
                init_input_q = True
                init_input_p = True

            # forward posterior
            means_q, logvars_q, hiddens_q, states_q = self.rnn_q(inputs_q, states_q, hiddens_p)

            # get experts
            # means_q : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_q, _logvars_q = self.experts(means_q[j], logvars_q[j], dim=2)
                means_q[j]   = _means_q
                logvars_q[j] = _logvars_q

            # sample z from posterior
            zs = self.rnn_q.sample(means_q, logvars_q)

            # aggregate observations (prior)
            if not init_input_p:
                #for m in range(self.num_multimodalities):
                #    # mod identifier
                #    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                #    # unpack input and combined contexts
                #    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                #    reps_context = pad_sequence(reps_context, context_sizes)
                #    reps_context = torch.sum(reps_context, dim=1)
                #    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                #    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                #    inputs_p += [input_p.unsqueeze(2)]

                ## concat inputs_q
                #inputs_p = torch.cat(inputs_p, dim=2)

                ## update flag
                #init_input_p = True
                raise ValueError

            # unsqueeze zs and expand it
            _zs = [z.unsqueeze(2).expand(num_episodes, self.nz, self.num_multimodalities, self.z_height, self.z_width)  for z in zs] 

            # forward prior
            _, means_p, logvars_p, hiddens_p, states_p = self.rnn_p(inputs_p, states_p, latents_q=_zs)

            # get experts
            # means_p : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_p, _logvars_p = self.experts(means_p[j], logvars_p[j], dim=2)
                means_p[j]   = _means_p
                logvars_p[j] = _logvars_p

            # append z to latent
            latents += [torch.cat(zs, dim=1).unsqueeze(1)] if len(zs) > 1 else [zs[0].unsqueeze(1)]

            # update accumulated KL
            for j in range(self.num_layers):
                loss_kl += loss_kld_gaussian_vs_gaussian(means_q[j], logvars_q[j], means_p[j], logvars_p[j])
                logprob_kl +=  logprob_gaussian(means_p[j],#.view(num_episodes, -1),
                                             logvars_p[j],#.view(num_episodes, -1),
                                             zs[j],#.view(num_episodes, -1),
                                             do_sum=False)
                logprob_kl += -logprob_gaussian(means_q[j],#.view(num_episodes, -1),
                                             logvars_q[j],#.view(num_episodes, -1),
                                             zs[j],#.view(num_episodes, -1),
                                             do_sum=False)

        ''' likelihood '''
        info = {}
        info['logprob_mod_likelihoods'] = []
        logprob_likelihood = 0
        info['mod_likelihoods'] = []
        loss_likelihood = 0
        mean_recons = []
        for idx, (dim, input_tuple) in enumerate(zip(self.dims, input_tuples)):
            channels, height, width, _, mtype = dim
            mod_target, mod_queries, mod_target_indices, mod_batch_sizes = input_tuple
            if len(mod_queries) > 0:# is not None:
                num_mod_data = len(mod_target)
                assert sum(mod_batch_sizes) == num_mod_data

                # run renderer (likelihood)
                mod_mean_recon = self._forward_renderer(idx, mod_queries, latents, num_episodes, mod_batch_sizes, mod_target_indices).detach()

                # convert to gray scale
                if mtype == 'image' and is_grayscale:
                    mod_mean_recon = rgb2gray(mod_mean_recon)
                    mod_target = rgb2gray(mod_target)
                    if not use_uint8:
                        mod_mean_recon = mod_mean_recon/255
                        mod_target = mod_target/255
                elif mtype == 'image' and use_uint8:
                    mod_mean_recon = 255*mod_mean_recon
                    mod_target = 255*mod_target

                # estimate recon loss
                loss_mod_likelihood = loss_recon_gaussian_w_fixed_var(mod_mean_recon, mod_target, std=std, add_logvar=False).detach()
                logprob_mod_likelihood = logprob_gaussian_w_fixed_var(
                            mod_mean_recon, #.view(num_episodes, -1),
                            mod_target, #.view(num_episodes, -1),
                            std=std,
                            do_sum=False).detach()

                # estimate recon loss without std
                loss_mod_likelihood_nostd = loss_recon_gaussian_w_fixed_var(mod_mean_recon.detach(), mod_target).detach()
                #logprob_mod_likelihood_nostd = logprob_gaussian_w_fixed_var(
                #            mod_mean_recon.detach(), #.view(num_episodes, -1),
                #            mod_target, #.view(num_episodes, -1),
                #            do_sum=False).detach()

                # sum per episode
                logprob_mod_likelihood = sum_tensor_per_episode(
                        logprob_mod_likelihood,
                        mod_batch_sizes,
                        mod_target_indices,
                        num_episodes)
            else:
                mod_mean_recon = reps_context.new_zeros(0, channels, height, width)
                loss_mod_likelihood = None
                loss_mod_likelihood_nostd = None
                logprob_mod_likelihood = None

            # add to loss_likelihood
            if loss_mod_likelihood is not None:
                loss_likelihood += loss_mod_likelihood
            if logprob_mod_likelihood is not None:
                logprob_likelihood += logprob_mod_likelihood

            # append to list
            mean_recons += [mod_mean_recon]
            info['mod_likelihoods'] += [loss_mod_likelihood_nostd]
            info['logprob_mod_likelihoods'] += [logprob_mod_likelihood]

        ''' loss '''
        # sum loss
        loss = loss_likelihood + beta * loss_kl
        logprob = logprob_likelihood + logprob_kl

        # additional loss info
        info['likelihood'] = loss_likelihood.detach()
        info['kl'] = loss_kl.detach()

        # return
        #return img_mean_recon, hpt_mean_recon, None, loss, info
        #return mean_recons, latents, loss, info
        return mean_recons, latents, logprob, info

    def infogain(self,
                 reps_context_sizes_pairs_c,
                 reps_context_sizes_pairs_t,
                 input_tuples,
                 num_steps=None, beta=1.0, std=1.0):
        # init
        num_episodes = len(reps_context_sizes_pairs_c[0][0])
        #assert len(set([index for _, _, mod_target_indices, _ in input_tuples for index in mod_target_indices])) == num_episodes
        loss_kl = 0
        weight = next(self.parameters())
        if self.modality_identifier.device != weight.device:
            self.modality_identifier = self.modality_identifier.to(weight.device)

        ''' forward posterior / prior '''
        # init states
        states_p  = self.rnn_p.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        states_q  = self.rnn_q.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        hiddens_p = [state_p[0] for state_p in states_p]
        #hiddens_q = [state_q[0] for state_q in states_q]
        latents = []
        inputs_q = []
        inputs_p = []
        init_input_q = False
        init_input_p = False
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # aggregate observations (posterior)
            if not init_input_q:
                for m in range(self.num_multimodalities):
                    # mod identifier
                    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                    # unpack input and combined contexts
                    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                    reps_context = pad_sequence(reps_context, context_sizes)
                    reps_context = torch.sum(reps_context, dim=1)
                    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                    inputs_p += [input_p.unsqueeze(2)]

                    reps_target, target_sizes = reps_context_sizes_pairs_t[m]
                    reps_target = pad_sequence(reps_target, target_sizes)
                    reps_target = torch.sum(reps_target, dim=1)
                    reps_target = reps_target.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_q = torch.cat([reps_target, reps_context, mod_identifier], dim=1)
                    inputs_q += [input_q.unsqueeze(2)]

                # concat inputs_q
                inputs_p = torch.cat(inputs_p, dim=2)

                # concat inputs_q
                inputs_q = torch.cat(inputs_q, dim=2)

                # update flag
                init_input_q = True
                init_input_p = True

            # forward posterior
            means_q, logvars_q, hiddens_q, states_q = self.rnn_q(inputs_q, states_q, hiddens_p)

            # get experts
            # means_q : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_q, _logvars_q = self.experts(means_q[j], logvars_q[j], dim=2)
                means_q[j]   = _means_q
                logvars_q[j] = _logvars_q

            # sample z from posterior
            zs = self.rnn_q.sample(means_q, logvars_q)

            # aggregate observations (prior)
            if not init_input_p:
                #for m in range(self.num_multimodalities):
                #    # mod identifier
                #    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                #    # unpack input and combined contexts
                #    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                #    reps_context = pad_sequence(reps_context, context_sizes)
                #    reps_context = torch.sum(reps_context, dim=1)
                #    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                #    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                #    inputs_p += [input_p.unsqueeze(2)]

                ## concat inputs_q
                #inputs_p = torch.cat(inputs_p, dim=2)

                ## update flag
                #init_input_p = True
                raise ValueError

            # unsqueeze zs and expand it
            _zs = [z.unsqueeze(2).expand(num_episodes, self.nz, self.num_multimodalities, self.z_height, self.z_width)  for z in zs] 

            # forward prior
            _, means_p, logvars_p, hiddens_p, states_p = self.rnn_p(inputs_p, states_p, latents_q=_zs)

            # get experts
            # means_p : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_p, _logvars_p = self.experts(means_p[j], logvars_p[j], dim=2)
                means_p[j]   = _means_p
                logvars_p[j] = _logvars_p

            # append z to latent
            latents += [torch.cat(zs, dim=1).unsqueeze(1)] if len(zs) > 1 else [zs[0].unsqueeze(1)]

            # update accumulated KL
            for j in range(self.num_layers):
                #loss_kl += loss_kld_gaussian_vs_gaussian(means_q[j], logvars_q[j], means_p[j], logvars_p[j], do_sum=False)
                loss_kl += logprob_gaussian(means_q[j],#.view(num_episodes, -1),
                                             logvars_q[j],#.view(num_episodes, -1),
                                             zs[j],#.view(num_episodes, -1),
                                             do_sum=False)
                loss_kl += -logprob_gaussian(means_p[j],#.view(num_episodes, -1),
                                             logvars_p[j],#.view(num_episodes, -1),
                                             zs[j],#.view(num_episodes, -1),
                                             do_sum=False)

        ''' loss '''
        # additional loss info
        info = {}
        info['kl'] = loss_kl.detach()

        # return
        #return img_mean_recon, hpt_mean_recon, None, loss, info
        return None, latents, loss_kl.detach(), info

    def infer(self,
              reps_context_sizes_pairs_c,
              num_steps=None,
              ):
        # init
        num_episodes = len(reps_context_sizes_pairs_c[0][0])
        weight = next(self.parameters())
        if self.modality_identifier.device != weight.device:
            self.modality_identifier = self.modality_identifier.to(weight.device)

        ''' forward posterior / prior '''
        # init states
        states_p  = self.rnn_p.init_state(num_episodes, [self.num_multimodalities, self.z_height, self.z_width]) # note: for training initial state, you need to change this (as the init_state should differ from each modality
        #hiddens_p = [state_p[0] for state_p in states_p]
        latents = []
        inputs_p = []
        init_input_p = False
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # forward prior (prob)
            means_p, logvars_p = self.rnn_p.forward_prob(states_p)

            # get experts
            # means_p : num_layers x (batch, zc, num_mod_data, zh, zw)
            #           -> num_layers x (num_mod_data, batch, zc, zh, zw)
            for j in range(self.num_layers):
                _means_p, _logvars_p = self.experts(means_p[j], logvars_p[j], dim=2)
                means_p[j]   = _means_p
                logvars_p[j] = _logvars_p

            # sample z from prior
            zs = self.rnn_p.sample(means_p, logvars_p)

            # aggregate observations (prior)
            if not init_input_p:
                for m in range(self.num_multimodalities):
                    # mod identifier
                    mod_identifier = self.modality_identifier[m:m+1, :, :, :].expand(num_episodes, self.num_multimodalities, self.z_height, self.z_width)

                    # unpack input and combined contexts
                    reps_context, context_sizes = reps_context_sizes_pairs_c[m]
                    reps_context = pad_sequence(reps_context, context_sizes)
                    reps_context = torch.sum(reps_context, dim=1)
                    reps_context = reps_context.view(-1, self.nc_context, self.z_height, self.z_width)

                    input_p = torch.cat([reps_context, mod_identifier], dim=1)
                    inputs_p += [input_p.unsqueeze(2)]

                # concat inputs_q
                inputs_p = torch.cat(inputs_p, dim=2)

                # update flag
                init_input_p = True

            # unsqueeze zs and expand it
            _zs = [z.unsqueeze(2).expand(num_episodes, self.nz, self.num_multimodalities, self.z_height, self.z_width)  for z in zs]

            # forward prior (rnn)
            hiddens_p, states_p = self.rnn_p.forward_rnn(inputs_p, states_p, _zs)

            # append z to latent
            latents += [torch.cat(zs, dim=1).unsqueeze(1)] if len(zs) > 1 else [zs[0].unsqueeze(1)]

        # return
        return latents

    def _forward_renderer(self, mod_idx, mod_queries, latents, num_episodes, mod_batch_sizes, mod_target_indices=[]):
        ''' forward image renderer '''
        # embed query
        emb_mod_queries = [self.query_encoders[mod_idx](mod_query) for mod_query in mod_queries]

        # concat z
        z = torch.cat(latents, dim=1)

        # run renderer (likelihood)
        #mod_mean_recon, mod_logvar_recon = self.renderers[0](z, emb_mod_queries, mod_batch_sizes)
        mean_recon = self.renderers[mod_idx](z, emb_mod_queries, mod_batch_sizes, mod_target_indices, num_episodes)
        return mean_recon#, logvar_recon

class CGQN(nn.Module):
    def __init__(self,
                 dims=[(3, 64, 64, 7, 'image'),
                       (132, 1, 1, 7, 'haptic')],
                 #im_height,  # image height
                 #im_channels,  # number of channels in image
                 #nc_img_query=7,  # kernel size (number of channels) for query
                 #hp_height=1,  # haptic height
                 #hp_channels=132,  # number of channels in haptic
                 #nc_hpt_query=7, #nc_hpt_query=20,  # kernel size (number of channels) for query
                 nc_enc=32,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nc_context=256,  # kernel size (number of channels) for representation
                 nz=3,  # size of latent variable
                 num_steps=4,  # number of steps in Draw
                 num_layers=1,  # number of StackedConvDrawEncoderCell/StackedConvDrawDecoderCell layers
                 ):
        super().__init__()
        # check conditions
        assert len(dims) > 0
        for dim in dims:
            assert len(dim) == 5, dim
            channels, height, width, nc_query, mtype = dim
            #assert height == width
            assert mtype in ['image', 'haptic']

        # init
        self.dims = dims
        self.num_multimodalities = len(dims)

        ## temporary
        #assert len(dims) == 2
        #im_height, im_channels, nc_img_query, mtype = dims[0]
        #hp_height, hp_channels, nc_hpt_query, mtype = dims[1]

        # define networks
        self.convdraw = CGQNConvDraw(
                dims,
                #im_height, im_channels, nc_img_query,
                #hp_height, hp_channels, nc_hpt_query,
                nc_enc, nc_lstm, nc_context,
                nz,
                num_steps,
                num_layers,
                )

        repnets = []
        for dim in dims:
            channels, height, width, nc_query, mtype = dim
            if mtype == 'image':
                repnets += [ImageContextNetwork(
                        nheight=height,
                        nwidth=width,
                        nchannels=channels,
                        nz=self.convdraw.nc_context,
                        train_init_representation=True,
                        )]
            elif mtype == 'haptic':
                repnets += [HapticContextNetwork(
                        nheight=height,
                        nchannels=channels,
                        nhidden = 512, #1024,
                        nz=self.convdraw.nc_context*self.convdraw.z_height*self.convdraw.z_width,
                        train_init_representation=True,
                        )]
            else:
                raise NotImplementedError
        self.repnets = nn.ModuleList(repnets)

    #def _forward(self, contexts, targets, num_steps=None, beta=1.0, std=1.0):
    def _forward(self, contexts, targets, merge_cxt_tgt=False):
        '''
        Input:
            contexts: a list, whose element is context
                      where context = (image, camera)
            targets:  a list, whose element is target
                      where target = (image, camera)
        Output:
            representations = batch_size x num_channels x num_height x num_width
        '''
        # init
        assert len(contexts[0]) == 2*self.num_multimodalities

        ''' run repnets (contexts) '''
        # run repnets (contexts)
        reps_context_sizes_pairs_c = []
        for i in range(self.num_multimodalities):
            # split contexts and targets
            mod_contexts = [(data_query_pairs[i*2], data_query_pairs[i*2+1]) for data_query_pairs in contexts] # (data, query) pair

            # get context representation
            mod_reps_c, _, mod_context_sizes_c = self.repnets[i](mod_contexts) # context of i-th modularity
            mod_reps_c = [mod_rep_c.view(-1, self.convdraw.nc_context, self.convdraw.z_height, self.convdraw.z_width) for mod_rep_c in mod_reps_c]

            # append to list
            reps_context_sizes_pairs_c += [(mod_reps_c, mod_context_sizes_c)]

        ## combine reps
        #reps_c, context_sizes_c = combine_reps(reps_context_sizes_pairs_c)

        ''' run repnets (targets) '''
        # run repnets (targets)
        if merge_cxt_tgt:
            contexts_and_targets = merge_two_batch(contexts, targets)
        else:
            contexts_and_targets = targets #merge_two_batch(contexts, targets)
        reps_context_sizes_pairs_t = []
        for i in range(self.num_multimodalities):
            # split contexts and targets
            mod_targets  = [(data_query_pairs[i*2], data_query_pairs[i*2+1]) for data_query_pairs in contexts_and_targets]

            # get context representation
            mod_reps_t, _, mod_context_sizes_t = self.repnets[i](mod_targets) # context of i-th modularity
            mod_reps_t = [mod_rep_t.view(-1, self.convdraw.nc_context, self.convdraw.z_height, self.convdraw.z_width) for mod_rep_t in mod_reps_t]

            # append to list
            reps_context_sizes_pairs_t += [(mod_reps_t, mod_context_sizes_t)]

        ## combine reps
        #reps_t, context_sizes_t = combine_reps(reps_context_sizes_pairs_t)

        ''' pre-processing targets '''
        # pre-processing targets
        data_queries_target_indices_batch_sizes = []
        for i in range(self.num_multimodalities):
            mod_data, mod_queries, mod_target_indices = [], [], []
            for idx, target in enumerate(targets):
                # unpack
                mod_datum, mod_query = target[i*2], target[i*2+1]
                #assert daum is not None or haptic is not None, 'empty target'

                # add targets
                mod_data    += [mod_datum]  if mod_datum is not None else []
                mod_queries += [mod_query]  if mod_datum is not None else []
                mod_target_indices += [idx] if mod_datum is not None else []
            assert len(mod_data)  == len(mod_queries)

            # concatenate
            mod_data = torch.cat(mod_data, dim=0) if len(mod_data) > 0 else None

            # get batch_sizes
            mod_batch_sizes = [targets[idx][i*2].size(0) for idx in mod_target_indices]

            # append to list
            data_queries_target_indices_batch_sizes += [(mod_data, mod_queries, mod_target_indices, mod_batch_sizes)]

        return (reps_context_sizes_pairs_c,
                reps_context_sizes_pairs_t,
                data_queries_target_indices_batch_sizes)

    def forward(self, contexts, targets, num_steps=None, beta=1.0, std=1.0, is_grayscale=False, do_sum=True):
        (reps_context_sizes_pairs_c,
         reps_context_sizes_pairs_t,
         data_queries_target_indices_batch_sizes) = self._forward(contexts, targets)

        # run conv-draw
        (mean_recons,
         latent,
         loss,
         info) = self.convdraw(
                 reps_context_sizes_pairs_c,
                 reps_context_sizes_pairs_t,
                 data_queries_target_indices_batch_sizes,
                 num_steps=num_steps, beta=beta, std=std, is_grayscale=is_grayscale, do_sum=do_sum)
        return mean_recons, latent, loss, info

    def predict(self, contexts, targets, num_steps=None, beta=1.0, std=1.0, is_grayscale=False, use_uint8=True):
        (reps_context_sizes_pairs_c,
         reps_context_sizes_pairs_t,
         data_queries_target_indices_batch_sizes) = self._forward(contexts, targets)

        # run conv-draw
        (mean_recons,
         latent,
         loss,
         info) = self.convdraw.predict(
                 reps_context_sizes_pairs_c,
                 reps_context_sizes_pairs_t,
                 data_queries_target_indices_batch_sizes,
                 num_steps=num_steps, beta=beta, std=std, is_grayscale=is_grayscale, use_uint8=use_uint8)
        return mean_recons, latent, loss, info

    def infogain(self, contexts, targets, num_steps=None, beta=1.0, std=1.0):
        (reps_context_sizes_pairs_c,
         reps_context_sizes_pairs_t,
         data_queries_target_indices_batch_sizes) = self._forward(contexts, targets)

        # run conv-draw
        (mean_recons,
         latent,
         loss,
         info) = self.convdraw.infogain(
                 reps_context_sizes_pairs_c,
                 reps_context_sizes_pairs_t,
                 data_queries_target_indices_batch_sizes,
                 num_steps=num_steps, beta=beta, std=std)
        return mean_recons, latent, loss, info

    def _generate(self, contexts):
        '''
        Input:
            contexts: a list, whose element is context
                      where context = (image, camera)
            targets:  a list, whose element is target
                      where target = (image, camera)
        Output:
            representations = batch_size x num_channels x num_height x num_width
        '''
        # init
        assert len(contexts[0]) == 2*self.num_multimodalities

        ''' run repnets (contexts) '''
        # run repnets (contexts)
        reps_context_sizes_pairs_c = []
        for i in range(self.num_multimodalities):
            # split contexts and targets
            mod_contexts = [(data_query_pairs[i*2], data_query_pairs[i*2+1]) for data_query_pairs in contexts] # (data, query) pair

            # get context representation
            mod_reps_c, _, mod_context_sizes_c = self.repnets[i](mod_contexts) # context of i-th modularity
            mod_reps_c = [mod_rep_c.view(-1, self.convdraw.nc_context, self.convdraw.z_height, self.convdraw.z_width) for mod_rep_c in mod_reps_c]

            # append to list
            reps_context_sizes_pairs_c += [(mod_reps_c, mod_context_sizes_c)]

        ## combine reps
        #reps_c, context_sizes_c = combine_reps(reps_context_sizes_pairs_c)
        return reps_context_sizes_pairs_c

    def generate(self, contexts, queries, num_steps=None, is_grayscale=False):
        reps_context_sizes_pairs_c = self._generate(contexts)

        ''' pre-processing targets '''
        # pre-processing img targets
        queries_batch_sizes = []
        for i in range(self.num_multimodalities):
            mod_queries = queries[i]

            # pre-processing img targets
            if mod_queries is not None:
                mod_batch_sizes = [mod_query.size(0) for mod_query in mod_queries]
            else:
                mod_batch_sizes = []
                mod_queries = [] #None

            # append to list
            queries_batch_sizes += [(mod_queries, mod_batch_sizes)]

        # run conv-draw
        #img_mean_recon, hpt_mean_recon, latent = self.convdraw.generate(
        mean_recons, latents = self.convdraw.generate(
                reps_context_sizes_pairs_c,
                #reps_c, context_sizes_c,
                queries_batch_sizes,
                #img_queries=img_queries, img_batch_sizes=img_batch_sizes,
                #hpt_queries=hpt_queries, hpt_batch_sizes=hpt_batch_sizes,
                num_steps=num_steps,
                is_grayscale=is_grayscale)
        #return img_mean_recon, hpt_mean_recon, latent
        return mean_recons, latents

    def infer(self, contexts, num_steps=None):
        reps_context_sizes_pairs_c = self._generate(contexts)

        # run conv-draw
        latents = self.convdraw.infer(
                reps_context_sizes_pairs_c,
                num_steps=num_steps)
        return latents


##############
class CGQN_v1(CGQN):
    def __init__(self,
                 dims,
                 nc_enc=32,  # kernel size (number of channels) for encoder
                 nc_lstm=32,  # kernel size (number of channels) for lstm
                 nc_context=256,  # kernel size (number of channels) for representation
                 nz=4,  # size of latent variable
                 num_steps=12,  # number of steps in Draw
                 num_layers=1,
                 ):
        super().__init__(
                 dims=dims,
                 nc_enc=nc_enc,
                 nc_lstm=nc_lstm,
                 nc_context=nc_context,
                 nz=nz,
                 num_steps=num_steps,
                 num_layers=num_layers,
                 )

class CGQN_v2(CGQN):
    def __init__(self,
                 dims,
                 nc_enc=32,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nc_context=256,  # kernel size (number of channels) for representation
                 nz=3,  # size of latent variable
                 num_steps=4,  # number of steps in Draw
                 num_layers=1,
                 ):
        super().__init__(
                 dims=dims,
                 nc_enc=nc_enc,
                 nc_lstm=nc_lstm,
                 nc_context=nc_context,
                 nz=nz,
                 num_steps=num_steps,
                 num_layers=num_layers,
                 )

class CGQN_v3(CGQN):
    def __init__(self,
                 dims,
                 nc_enc=32,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nc_context=256,  # kernel size (number of channels) for representation
                 nz=3,  # size of latent variable
                 num_steps=8,  # number of steps in Draw
                 num_layers=1,
                 ):
        super().__init__(
                 dims=dims,
                 nc_enc=nc_enc,
                 nc_lstm=nc_lstm,
                 nc_context=nc_context,
                 nz=nz,
                 num_steps=num_steps,
                 num_layers=num_layers,
                 )

class CGQN_v4(CGQN):
    def __init__(self,
                 dims,
                 nc_enc=32,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nc_context=256,  # kernel size (number of channels) for representation
                 nz=3,  # size of latent variable
                 num_steps=12,  # number of steps in Draw
                 num_layers=1,
                 ):
        super().__init__(
                 dims=dims,
                 nc_enc=nc_enc,
                 nc_lstm=nc_lstm,
                 nc_context=nc_context,
                 nz=nz,
                 num_steps=num_steps,
                 num_layers=num_layers,
                 )
