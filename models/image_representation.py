import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import pack_sequence, unpack_sequence
from utils import loss_kld_gaussian_vs_gaussian, loss_recon_gaussian_w_fixed_var
from models.reparam import NormalDistributionConv2d, NormalDistributionConvTranspose2d


class SimpleCameraEncoder(nn.Module):
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

class CameraEncoder(nn.Module):
    ''' 
    reference:
    https://github.com/deepmind/gqn-datasets
    http://science.sciencemag.org/content/360/6394/1204
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, -1, 1, 1)

        # slice
        position = input[:, :3,  :, :]
        yaw      = input[:, 3:4, :, :]
        pitch    = input[:, 4:,  :, :]

        # transform
        output = torch.cat([position,
                            torch.cos(yaw),
                            torch.sin(yaw),
                            torch.cos(pitch),
                            torch.sin(pitch)], dim=1)
        return output

class ResidualConv3x3(nn.Module):
    '''
    references:
    1) https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    2) https://github.com/ogroth/tf-gqn/blob/master/gqn/gqn_encoder.py
    '''
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        # main path
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

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
    def __init__(self, nheight=64, nwidth=64, nc=3, nz=256, use_instance_norm=False):
        super().__init__()
        #assert nheight % 16 == 0, "nheight has to be a multiple of 16"
        #assert nheight == 64, "nheight has to be 64"
        assert nheight % 4 == 0, "nheight has to be a multiple of 4"

        # init
        self.nheight = nheight
        self.nwidth = nwidth
        self.nc    = nc
        self.nz    = nz
        self.use_instance_norm = use_instance_norm

        # define networks
        self.img_enc = nn.Sequential(
                nn.Conv2d(self.nc, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                ResidualConv3x3(256, 128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                )
        self.camera_enc = CameraEncoder()
        self.enc = nn.Sequential(
                ResidualConv3x3(256 + 7, 128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                )
        if self.use_instance_norm:
            self.norm = nn.InstanceNorm2d(self.nz)

    def get_output_size(self):
        #return self.nz, 16, 16
        return self.nz, self.nheight//4, self.nwidth//4

    def forward(self, image, camera):
        # encode image and camera
        emb_img = self.img_enc(image)
        emb_cm  = self.camera_enc(camera)

        # concat emb_img and emb_cm
        height, width = emb_img.size(2), emb_img.size(3)
        concat = torch.cat([emb_img, emb_cm.expand(emb_cm.size(0), emb_cm.size(1), height, width)], dim=1)

        # encode concat
        output = self.enc(concat)

        # unit norm output
        if self.use_instance_norm:
            output = self.norm(output)

        return output

class PoolRepresentationNetwork(TowerRepresentationNetwork):
    def __init__(self, isize=64, nc=3, nz=256, use_instance_norm=False):
        super().__init__(isize, nc, nz, use_instance_norm)
        self.pool = nn.AvgPool2d(16)

    def get_output_size(self):
        return self.nz, 1, 1

    def forward(self, image, camera):
        output = super().forward(image, camera)
        output = self.pool(output)
        return output

class ContextNetwork(nn.Module):
    def __init__(self,
                 nheight=64,
                 nwidth=64,
                 nchannels=3,
                 nhidden=256,
                 nz=256,
                 train_init_representation=False,
                 use_mean=False,
                 use_instance_norm=False,
                 ):
        super().__init__()
        # init
        self.train_init_representation = train_init_representation
        self.use_mean = use_mean
        self.use_instance_norm = use_instance_norm

        # define networks
        self.repnet = TowerRepresentationNetwork(
                nheight=nheight,
                nwidth=nwidth,
                nc=nchannels,
                nz=nz,
                use_instance_norm=use_instance_norm,
                )  # repnet=PoolRepresentationNetwork()

        # information from representation network
        num_hidden, height, width = self.repnet.get_output_size()
        self.num_hidden = num_hidden
        self.height = height
        self.width = width

        # initial representation
        if self.train_init_representation:
            self.init_representation = Parameter(torch.zeros(1, self.num_hidden, self.height, self.width))
            self.init_query = Parameter(torch.zeros(1, 7, 1, 1))

        # init pad
        self.camera_enc = CameraEncoder()

    def get_init_representation(self, batch_size):
        if self.train_init_representation:
            return self.init_representation.expand(batch_size, self.num_hidden, self.height, self.width)
        else:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, self.num_hidden, self.height, self.width)

    def get_init_query(self, batch_size):
        if self.train_init_representation:
            return self.init_query.expand(batch_size, 7, 1, 1)
        else:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, 7, 1, 1)

    def forward(self, contexts):
        '''
        Input:
            contexts: a list, whose element is context
                      where context = (image, camera)
        Output:
            representations = num_episodes x num_channels x num_height x num_width
        '''
        # init
        num_episodes = len(contexts)
        context_sizes = [context[0].size(0) if context[0] is not None else 0 for context in contexts]
        non_zero_context_sizes = [context[0].size(0) for context in contexts if context[0] is not None]
        with_init_context_sizes = [context[0].size(0)+1 if context[0] is not None else 1 for context in contexts] # including learnable initial representation

        # init representation
        images = []
        cameras = []

        # get images list and cameras list
        for i, context in enumerate(contexts):
            # unpack context
            image, camera = context

            # forward representation network (each context)
            if image is not None:
                images += [image]
                cameras += [camera]
        assert len(images) == len(cameras)

        # forward representation networks with pack_sequence
        if len(images) > 0:
            # pack_sequence
            packed_images = pack_sequence(images)
            packed_cameras = pack_sequence(cameras)

            # forward representation networks
            packed_representations = self.repnet(packed_images, packed_cameras)
            packed_keys = self.camera_enc(packed_cameras)

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

        ## init representation
        #representations = []

        ## run representation network
        #for context in contexts:
        #    # unpack context
        #    image, camera = context

        #    # add init_representation
        #    init_representation = self.get_init_representation(1)

        #    # forward representation network (each context)
        #    if image is None:
        #        # pass if context is empty
        #        representation = init_representation
        #    else:
        #        representation = self.repnet(image, camera)
        #        representation = torch.cat([representation, init_representation], dim=0)

        #    # sum over batch
        #    if self.use_mean:
        #        representation = torch.mean(representation, dim=0, keepdim=True)
        #    else:
        #        representation = torch.sum(representation, dim=0, keepdim=True)

        #    # append to representations
        #    representations += [representation]

        ## concat representations
        #representations = torch.cat(representations, dim=0) if len(representations) > 0 else 0

        #return representations
