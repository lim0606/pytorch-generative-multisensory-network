import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.convdraw import ConvLSTMCell, StackedConvDrawEncoderCell, StackedConvDrawDecoderCell, ConvLSTM
from models.reparam import NormalDistributionConv2d, NormalDistributionConvTranspose2d

from utils import broadcast_representation


class ImageRendererDecoder(nn.Module):
    '''
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    '''
    def __init__(self, im_channels, h_dim, lstm_dim):
        super().__init__()
        self.decode = nn.Conv2d(lstm_dim, h_dim, 5, stride=1, padding=2)
        self.convt = nn.ConvTranspose2d(h_dim, h_dim*2, 4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(h_dim*2, im_channels, 4, stride=2, padding=1)
        #self.reparam = NormalDistributionConvTranspose2d(h_dim*2, im_channels, kernel_size=4, stride=2, padding=1)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def forward(self, h):
        xx = F.relu(self.decode(h))
        xx = F.relu(self.convt(xx))
        #mu, logvar = self.reparam(xx)
        #return mu, logvar
        return self.convt2(xx)

class ImageRendererEncoder(nn.Module):
    '''
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    '''
    def __init__(self, im_channels, h_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(im_channels, h_dim*2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(h_dim*2, h_dim, 4, stride=2, padding=1)

    def forward(self, x):
        hidden = F.relu(self.conv1(x))
        return F.relu(self.conv2(hidden))

class ImageRenderer(nn.Module):
    '''
    Modified ConvDraw for GQN
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    in generator, emb_recon is discarded unlike original conv-draw 
    '''
    def __init__(self,
                 im_height=64,  # image height
                 im_width=64,  # image width
                 im_channels=3,  # number of channels in image
                 nc_query=7,  # kernel size (number of channels) for query
                 nc_enc=128,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nz_per_step=3,  # size of latent variable
                 z_num_steps=4, # number of steps in convlstm posterior/prior (i.e. DRAW's num_steps)
                 z_height=8,  # height of feature map size for z
                 z_width=8,  # width of feature map size for z
                 num_steps=4,  # number of steps in Renderer (no need to be the same size as DRAW's num_steps)
                 use_canvas_in_prior=False,  # canvas usage in prior (gqn paper didn't use canvas)
                 concat_z=True,
                 do_sum=True,
                 ):
        super().__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.nc_query = nc_query
        self.nc_enc = nc_enc
        self.nc_lstm = nc_lstm
        self.nz_per_step = nz_per_step
        self.z_num_steps = z_num_steps
        self.z_height = z_height
        self.z_width = z_width
        self.num_steps = num_steps
        self.use_canvas_in_prior = use_canvas_in_prior
        self.concat_z = concat_z
        self.do_sum = do_sum

        # get nz
        if self.concat_z:
            nz = nz_per_step * z_num_steps
        else:
            nz = nz_per_step
            assert self.num_steps == self.z_num_steps

        # define networks
        if self.use_canvas_in_prior:
            self.encoder = ImageRendererEncoder(im_channels, nc_enc)
            self.rnn_p = ConvLSTMCell(nc_enc + nz + nc_query, nc_lstm)
        else:
            self.rnn_p = ConvLSTMCell(nz + nc_query, nc_lstm)
        self.decoder = ImageRendererDecoder(
                im_channels,
                nc_enc,
                nc_lstm,
                )

    def forward(self, z, emb_queries, batch_sizes, indices, num_episodes, num_steps=None):
        # init
        #num_episodes = len(batch_sizes)
        num_data = sum(batch_sizes)
        num_steps = num_steps if num_steps is not None else self.num_steps

        # concat queries
        emb_query = torch.cat(emb_queries, dim=0)
        emb_query = emb_query.expand(emb_query.size(0), emb_query.size(1), self.z_height, self.z_width)

        # broadcast z
        z_sizes = list(z.size())
        z = broadcast_representation(z, num_episodes, batch_sizes, indices)
        assert z.size(0) == num_data

        # concat z
        if self.concat_z:
            z = z.view(num_data, z_sizes[1]*z_sizes[2], self.z_height, self.z_width)

        # init recon haptic
        if self.do_sum:
            mean_recon = z.new_zeros(num_data, self.im_channels, self.im_height, self.im_width)
            #logvar_recon = z.new_zeros(num_data, self.im_channels, self.im_height, self.im_width)

        # init states
        state_p = self.rnn_p.init_state(num_data, [self.z_height, self.z_width])
        hidden_p = state_p[0]
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # select z
            _z = z if self.concat_z else z[:, i, :, :, :]

            # update rnn
            if self.use_canvas_in_prior:
                emb_recon = self.encoder(mean_recon)
                input_p = torch.cat([_z, emb_recon, emb_query], dim=1)
            else:
                input_p = torch.cat([_z, emb_query], dim=1)
            hidden_p, state_p = self.rnn_p(input_p, state_p)

            # update recon
            if self.do_sum:
                #dmean_recon, dlogvar_recon = self.decoder(hidden_p)
                dmean_recon = self.decoder(hidden_p)
                mean_recon = mean_recon + dmean_recon
                #logvar_recon = logvar_recon + dlogvar_recon
            elif (i+1) == num_steps:
                mean_recon = self.decoder(hidden_p)

        # return
        return mean_recon
