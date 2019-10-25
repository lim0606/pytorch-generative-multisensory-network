import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.convdraw import ConvLSTMCell, StackedConvDrawEncoderCell, StackedConvDrawDecoderCell, ConvLSTM
from models.reparam import NormalDistributionConv2d, NormalDistributionConvTranspose2d

from utils import broadcast_representation


class HapticRendererDecoder(nn.Module):
    def __init__(self, input_dim=132, h_dim=400, lstm_dim=20, z_height=16, z_width=16):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.lstm_dim = lstm_dim
        self.z_height = z_height
        self.z_width = z_width

        self.fc1 = nn.Linear(lstm_dim*z_height*z_width, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, input_dim)
        #self.reparam = NormalDistributionLinear(h_dim, input_dim, nonlinearity='hard')  # clip logvar values
        #self.reparam = NormalDistributionLinear(h_dim, input_dim, nonlinearity='tanh')  # clip logvar values
        #self.reparam = NormalDistributionLinear(h_dim, input_dim, nonlinearity='2tanh')  # clip logvar values

    def forward(self, h):
        # init
        batch_size = h.size(0)
        input = h.view(batch_size, -1)

        # forward
        h1 = F.relu(self.fc1(input))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)

        return output
        #mu, logvar = self.reparam(h3)
        #return mu, logvar

class HapticRenderer(nn.Module):
    '''
    Modified ConvDraw for GQN
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    in generator, emb_recon is discarded unlike original conv-draw
    '''
    def __init__(self,
                 hp_height=64,  # image height
                 hp_width=64,  # image width
                 hp_channels=3,  # number of channels in image
                 nc_query=7,  # kernel size (number of channels) for query
                 nc_enc=128,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nz_per_step=3,  # size of latent variable
                 z_num_steps=4, # number of steps in convlstm posterior/prior (i.e. DRAW's num_steps)
                 z_height=8,  # height of feature map size for z 
                 z_width=8,  # width of feature map size for z 
                 num_steps=4,  # number of steps in Renderer (no need to be the same size as Draw's num_steps)
                 use_canvas_in_prior=False,  # canvas usage in prior (gqn paper didn't use canvas)
                 concat_z=True,
                 do_sum=True,
                 ):
        super().__init__()
        self.hp_height   = hp_height
        self.hp_width   = hp_width
        self.hp_channels = hp_channels
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
            raise NotImplementedError
            #self.encoder = ImageRendererEncoder(im_channels, nc_enc)
            self.rnn_p = ConvLSTMCell(nc_enc + nz + nc_query, nc_lstm)
        else:
            self.rnn_p = ConvLSTMCell(nz + nc_query, nc_lstm)
        self.decoder = HapticRendererDecoder(
                input_dim = hp_channels*hp_height*hp_width,
                h_dim = 512, #1024,
                lstm_dim = nc_lstm,
                z_height = z_height,
                z_width = z_width,
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
            mean_recon = z.new_zeros(num_data, self.hp_channels*self.hp_height*self.hp_width)
            #logvar_recon = z.new_zeros(num_data, self.hpm_channels*self.hpm_height*self.hp_width)

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
