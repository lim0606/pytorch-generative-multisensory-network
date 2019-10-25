import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import pack_sequence, unpack_sequence
from utils import loss_kld_gaussian_vs_gaussian, loss_recon_gaussian_w_fixed_var
from models.reparam import NormalDistributionConv2d, NormalDistributionConvTranspose2d
#from models.attention_layers import SelfAttention

class SimpleHandEncoder(nn.Module):
    '''
    reference:
    https://github.com/deepmind/gqn-datasets
    http://science.sciencemag.org/content/360/6394/1204
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size, -1, 1, 1)
        return output

class HandEncoder(nn.Module):
    ''' 
    reference:
    https://github.com/deepmind/gqn-datasets
    http://science.sciencemag.org/content/360/6394/1204
    '''
    def __init__(self, use_pose=False):
        super().__init__()
        self.use_pose = use_pose

    def forward(self, input):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, -1, 1, 1)

        if self.use_pose:
            # slice
            pose     = input[:, :13,  :, :]
            position = input[:, 13:16,  :, :]
            pitch    = input[:, 16:17, :, :]
            yaw      = input[:, 17:,  :, :]

            # transform
            output = torch.cat([pose,
                                position,
                                torch.cos(pitch),
                                torch.sin(pitch),
                                torch.cos(yaw),
                                torch.sin(yaw)], dim=1)
        else:
            # slice
            position = input[:, :3,  :, :]
            pitch    = input[:, 3:4, :, :]
            yaw      = input[:, 4:,  :, :]

            # transform
            output = torch.cat([position,
                                torch.cos(pitch),
                                torch.sin(pitch),
                                torch.cos(yaw),
                                torch.sin(yaw)], dim=1)

        return output

class ResidualConv1x1(nn.Module):
    '''
    references:
    1) https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    2) https://github.com/ogroth/tf-gqn/blob/master/gqn/gqn_encoder.py
    '''
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        # main path
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

        # skip connection
        if in_channels != out_channels: 
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        else:
            self.skip = None

    def forward(self, input):
        # conv path
        output = self.main(input)

        # skip connection
        if self.skip is not None: 
            x = self.skip(input)
        else:
            x = input

        # add
        output += x

        return output

class TowerRepresentationNetwork(nn.Module):
    def __init__(self, isize=1, nc=132, nhidden=256, nz=256, normalize_input=False, use_pose=False, use_instance_norm=False, output_height=None, output_width=None):
        super().__init__()
        #assert isize % 16 == 0, "isize has to be a multiple of 16"
        assert isize == 1, "isize has to be 64"

        # init
        self.isize = isize
        self.nc    = nc
        self.nz    = nz
        self.normalize_input = normalize_input
        self.use_pose = use_pose
        self.nc_query = 20 if self.use_pose else 7
        self.use_instance_norm = use_instance_norm
        self.output_height = output_height
        self.output_width = output_width

        # define networks
        self.ppc_enc = nn.Sequential(
                nn.Conv2d(self.nc, nhidden, kernel_size=1, stride=1),
                nn.ReLU(),
                ResidualConv1x1(nhidden, nhidden // 2),
                nn.ReLU(),
                nn.Conv2d(nhidden // 2, nhidden, kernel_size=1, stride=1),
                nn.ReLU(),
                )
        self.hand_enc = HandEncoder()
        self.enc = nn.Sequential(
                ResidualConv1x1(nhidden + self.nc_query, nhidden // 2),
                nn.ReLU(),
                nn.Conv2d(nhidden // 2, nhidden, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(nhidden, self.nz, kernel_size=1, stride=1),
                nn.ReLU(),
                )
        if self.use_instance_norm:
            self.norm = nn.InstanceNorm2d(self.nz//(self.output_height*self.output_width))

        # load stat
        if self.normalize_input:
            #with open('cache/shepard_metzler_5_parts_stochastic_stat.pt', 'rb') as f:
            #with open('cache/shepard_metzler_5_parts_semi_det_stat.pt', 'rb') as f:
            with open('cache/shepard_metzler_5_parts_stat.pt', 'rb') as f:
                stat = torch.load(f)
                self.mean = stat['mean'].view(1, 132)[:, :self.nc]
                #self.mean[:, 113:] = 0
                self.std = stat['std'].view(1, 132)[:, :self.nc]

    def get_output_size(self):
        return self.nz, 1, 1

    def forward(self, ppceptive, hand):
        # init
        ppceptive = ppceptive.view(-1, self.nc)

        # normalize input
        if self.normalize_input:
            if self.mean.device != ppceptive.device:
                self.mean = self.mean.to(ppceptive.device)
                self.std = self.std.to(ppceptive.device)
            ppceptive = (ppceptive - self.mean) / (self.std + 1e-8)

        # reshape input
        batch_size = ppceptive.size(0)
        ppceptive = ppceptive.view(batch_size, -1, 1, 1)

        # encode ppceptive and hand
        emb_ppc = self.ppc_enc(ppceptive)
        emb_cm  = self.hand_enc(hand)

        # concat emb_ppc and emb_cm
        height, width = emb_ppc.size(2), emb_ppc.size(3)
        concat = torch.cat([emb_ppc, emb_cm.expand(emb_cm.size(0), emb_cm.size(1), height, width)], dim=1)

        # encode concat
        output = self.enc(concat)

        # unit norm output
        if self.use_instance_norm:
            output = output.view(batch_size, -1, self.output_height, self.output_width)
            output = self.norm(output)
            output = output.view(batch_size, -1, 1, 1)

        return output

class ContextNetwork(nn.Module):
    def __init__(self,
                 nheight=1,
                 nchannels=132,
                 nhidden=256,
                 nz=256,
                 output_height=None,
                 output_width=None,
                 train_init_representation=False,
                 normalize_input=False,
                 use_pose=False,
                 use_instance_norm=False,
                 ):
        super().__init__()
        # init
        self.train_init_representation = train_init_representation
        self.normalize_input = normalize_input
        self.use_pose = use_pose
        self.nquery = 20 if self.use_pose else 7
        self.use_instance_norm = use_instance_norm
        self.output_height = output_height
        self.output_width = output_width

        # define networks
        self.repnet = TowerRepresentationNetwork(
                isize=nheight,
                nc=nchannels,
                nhidden=nhidden,
                nz=nz,
                output_height=output_height,
                output_width=output_width,
                normalize_input=normalize_input,
                use_pose=use_pose,
                use_instance_norm=use_instance_norm,
                )  # repnet=PoolRepresentationNetwork(),

        # information from representation network
        num_reps, height, width = self.repnet.get_output_size()
        self.num_reps = num_reps
        self.height = height
        self.width = width

        # initial representation
        if self.train_init_representation:
            self.init_representation = Parameter(torch.zeros(1, self.num_reps, self.height, self.width))
            self.init_query = Parameter(torch.zeros(1, self.nquery, 1, 1))

        # init pad
        self.hand_enc = HandEncoder(use_pose=use_pose)
        #self.pad = torch.zeros(1, self.num_reps, self.height, self.width)
        #self.kpad = torch.zeros(1, 20)

        ## define self attention network
        #self.attention = SelfAttention(
        #        input_dim=nz,
        #        key_dim=512,
        #        value_dim=nz,
        #        hidden_dim=512,
        #        )

    def get_init_representation(self, batch_size):
        if self.train_init_representation:
            return self.init_representation.expand(batch_size, self.num_reps, self.height, self.width)
        else:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, self.num_reps, self.height, self.width)

    def get_init_query(self, batch_size):
        if self.train_init_representation:
            return self.init_query.expand(batch_size, self.nquery, 1, 1)
        else:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, self.nquery, 1, 1)

    def forward(self, contexts):
        '''
        Input:
            contexts: a list, whose element is context
                      where context = (ppceptive, hand)
        Output:
            representations = num_episodes x num_channels x num_height x num_width
        '''
        # init
        num_episodes = len(contexts)
        context_sizes = [context[0].size(0) if context[0] is not None else 0 for context in contexts]
        non_zero_context_sizes = [context[0].size(0) for context in contexts if context[0] is not None]
        with_init_context_sizes = [context[0].size(0)+1 if context[0] is not None else 1 for context in contexts] # including learnable initial representation

        # init representation
        ppceptives = []
        hands = []

        # get ppceptives list and hands list
        for i, context in enumerate(contexts):
            # unpack context
            ppceptive, hand = context

            # forward representation network (each context)
            if ppceptive is not None:
                ppceptives += [ppceptive]
                hands += [hand]
        assert len(ppceptives) == len(hands)

        # forward representation networks with pack_sequence
        if len(ppceptives) > 0:
            # pack_sequence
            packed_ppceptives = pack_sequence(ppceptives)
            packed_hands = pack_sequence(hands)

            # forward representation networks
            packed_representations = self.repnet(packed_ppceptives, packed_hands)
            packed_keys = self.hand_enc(packed_hands)

            # unpack sequence
            _representations = unpack_sequence(packed_representations, non_zero_context_sizes)
            _keys = unpack_sequence(packed_keys, non_zero_context_sizes)

        # concat init_representation
        representations = []
        keys = []
        idx = 0
        for i, context_size in enumerate(context_sizes):
            # get init representation
            init_representation = self.get_init_representation(1)
            init_key = self.get_init_query(1)

            # get representation
            if context_size > 0:
                representation = torch.cat([
                    init_representation,
                    _representations[idx],
                    ], dim=0)
                key = torch.cat([
                    init_key,
                    _keys[idx],
                    ], dim=0)
                idx = idx + 1
            else:
                # set to representation
                representation = init_representation
                key = init_key

            # append to list
            representations += [representation]
            keys += [key]
        assert idx == len(non_zero_context_sizes)

        return representations, keys, with_init_context_sizes
