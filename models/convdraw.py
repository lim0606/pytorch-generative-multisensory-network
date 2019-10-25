'''
copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.reparam import NormalDistributionConv2d
from utils import pack_hiddens, unpack_hiddens


class ConvLSTMCell(nn.Module):
    '''
    Generate a convolutional LSTM cell
    copied and modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    '''
    def __init__(self, input_size, hidden_size, kernel_size=5, stride=1, padding=2, train_init_state=False, height=None, width=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.train_init_state = train_init_state
        self.height = height
        self.width = width

        # lstm gates
        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)

        # initial states
        if self.train_init_state:
            assert self.height and self.width
            self.init_hidden = Parameter(torch.zeros(1, self.hidden_size, self.height, self.width))
            self.init_cell   = Parameter(torch.zeros(1, self.hidden_size, self.height, self.width))

    def init_state(self, batch_size, spatial_size):
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        if self.train_init_state:
            return (self.init_hidden.expand(state_size),
                    self.init_cell.expand(state_size))
        else:
            weight = next(self.parameters())
            return (weight.new_zeros(state_size),
                    weight.new_zeros(state_size))

    def forward(self, input, prev_state):
        ''' forward stacked rnn one time step
        Input:
            input: batch_size x input_size x height x width
            prev_state: (hidden, cell) of each ConvLSTM
        Output:
            output: hidden (of new_state), batch_size x hidden_size x hidden_height x hidden_width
            new_state: (hidden, cell) of each ConvLSTM
        '''

        # get batch and spatial sizes
        batch_size = input.data.size(0)
        spatial_size = input.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(batch_size, spatial_size)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input, prev_hidden), 1)
        outputs = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = outputs.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        # pack output
        new_state = (hidden, cell)
        output = hidden
        return output, new_state

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=5, stride=1, padding=2, train_init_state=False, height=None, width=None, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.train_init_state = train_init_state
        self.height = height
        self.width = width
        self.num_layers = num_layers
        if self.num_layers != 1:
            raise NotImplementedError

        # define rnn
        self.rnn = ConvLSTMCell(input_size, hidden_size, kernel_size, stride, padding, train_init_state, height, width)

        # initial states
        if self.train_init_state:
            assert self.height and self.width
            self.init_hidden = Parameter(torch.zeros(1, 1, self.hidden_size, self.height, self.width))
            self.init_cell   = Parameter(torch.zeros(1, 1, self.hidden_size, self.height, self.width))

    def init_state(self, batch_size, spatial_size):
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        if self.train_init_state:
            return (self.init_hidden.expand(state_size),
                    self.init_cell.expand(state_size))
        else:
            weight = next(self.parameters())
            return (weight.new_zeros(state_size),
                    weight.new_zeros(state_size))

    def forward(self, input, hiddens):
        ''' forward stacked rnn with sequence input
        Input:
            input: tensor, seq_len x batch_size x input_size
            hiddens: (h_t, c_t) = a tuple if lstm
                     h_t = a tensor else
                     where h_t (and c_t) is tensor num_layers x batch_size x hidden_size x height x width

        Output:
            output: tensor, seq_len x batch_size x output_size
            hiddens: (h_t, c_t) = a tuple if lstm
                     h_t = a tensor else
                     where h_t (and c_t) is tensor num_layers x batch_size x hidden_size x height x width
        '''
        # init output
        output = []

        # unpack hiddens
        hiddens = unpack_hiddens(hiddens, 'LSTM')

        # forward per time step
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            # forward one time step
            output_t, hiddens = self.rnn(input_t, hiddens)

            # append output_t to output
            output += [output_t.unsqueeze(0)]

        # concat output
        output = torch.cat(output, dim=0)

        # pack hiddens
        hiddens = pack_hiddens(hiddens, 'LSTM')

        return output, hiddens

    def flatten_parameters(self):
        pass

class ConvDrawEncoderCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,  # i.e. nc_lstm
                 nz,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 ):
        super().__init__()
        self.rnn = ConvLSTMCell(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.reparam = NormalDistributionConv2d(hidden_size, nz, kernel_size=kernel_size, stride=stride, padding=padding)

    def init_state(self, batch_size, spatial_size):
        return self.rnn.init_state(batch_size, spatial_size)

    def sample(self, mean, logvar):
        return self.reparam.sample_gaussian(mean, logvar)

    def forward(self, input, prev_state):
        ''' forward stacked rnn one time step
        Input:
            input:      batch_size x input_size x height x width
            prev_state: (hidden, cell) of each ConvLSTM
        Output:
            mean:      batch_size x hidden_size x hidden_height x hidden_width
            logvar:    batch_size x hidden_size x hidden_height x hidden_width
            hidden:    hidden (of new_state), batch_size x hidden_size x hidden_height x hidden_width
            new_state: (hidden, cell) of each ConvLSTM
        '''

        # forward rnn
        hidden, new_state = self.rnn(input, prev_state)

        # forward mean, logvar
        mean, logvar = self.reparam(hidden)

        return mean, logvar, hidden, new_state

class StackedConvDrawEncoderCell(nn.Module):
    def __init__(self,
                 input_size,
                 #context_size,  # i.e. nc_context
                 hidden_size,  # i.e. nc_lstm
                 nz,
                 kernel_size=5,
                 padding=2,
                 num_layers=1,
                 dropout=0,
                 ):
        super().__init__()
        self.input_size = input_size
        #self.context_size = context_size
        self.hidden_size = hidden_size
        self.nz = nz
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers
        self.dropout = dropout

        rnns = []
        #rnns.append(ConvDrawEncoderCell(input_size+hidden_size+context_size, hidden_size, nz))
        rnns.append(
                ConvDrawEncoderCell(input_size+hidden_size,
                                    hidden_size,
                                    nz,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    ))
        for i in range(1, num_layers):
            #rnns.append(ConvDrawEncoderCell(nz+hidden_size+context_size, hidden_size, nz))
            rnns.append(
                    ConvDrawEncoderCell(nz+hidden_size,
                                        hidden_size,
                                        nz,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        ))
        self.rnns = nn.ModuleList(rnns)

    def init_state(self, batch_size, spatial_size):
        states = []
        for i in range(len(self.rnns)):
            state = self.rnns[i].init_state(batch_size, spatial_size)
            states += [state]
        return states

    def sample(self, means, logvars):
        zs = []
        for i, (mean, logvar) in enumerate(zip(means, logvars)):
            z = self.rnns[i].sample(mean, logvar)
            zs += [z]
        return zs

    #def forward(self, input, context, prev_states, dec_hiddens):
    def forward(self, input, prev_states, dec_hiddens):
        ''' forward stacked rnn one time step⋅
        Input:
            input: batch_size x input_size
            prev_states: a list with length = number of layers (of stacked RNNs)
                     each element in the list: (h_t, c_t) (or h_t) of each RNN
        Output:
            output_t: batch_size x output_size
            new_states: a list with length = number of layers (of stacked RNNs)
                         each element in the list: (h_t, c_t) (or h_t) of each RNN
        '''
        # init new_states
        new_means = []
        new_logvars = []
        new_hiddens = []
        new_states = []

        # init input (first layer)
        hidden_p = dec_hiddens[0]
        #input_q = torch.cat([input, hidden_p, context], dim=1)
        input_q = torch.cat([input, hidden_p], dim=1)

        # forward rnn (first layer)
        mean_q, logvar_q, hidden_q, state = self.rnns[0](input_q, prev_states[0])
        new_means += [mean_q]
        new_logvars += [logvar_q]
        new_hiddens += [hidden_q]
        new_states += [state]

        # remaining layers
        for j in range(1, self.num_layers):
            # init input
            hidden_p = dec_hiddens[j]
            #input_q = torch.cat([mean_q, hidden_p, context], dim=1)
            input_q = torch.cat([mean_q, hidden_p], dim=1)

            # apply dropout
            ''' see https://discuss.pytorch.org/t/lstm-dropout-clarification-of-last-layer/5588 '''
            if self.dropout > 0:
                input_q = F.dropout(input_q, p=self.dropout, training=self.training, inplace=False)

            # forward rnn
            mean_q, logvar_q, hidden_q, state = self.rnns[j](input_q, prev_states[j])
            new_means += [mean_q]
            new_logvars += [logvar_q]
            new_hiddens += [hidden_q]
            new_states += [state]

        return new_means, new_logvars, new_hiddens, new_states

class ConvDrawDecoderCell(nn.Module):
    def __init__(self,
                 lstm_input_size,
                 reparam_input_size,
                 hidden_size,  # i.e. nc_lstm
                 nz,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 dropout=0,
                 ):
        super().__init__()
        self.rnn_p = ConvLSTMCell(lstm_input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.reparam_p = NormalDistributionConv2d(reparam_input_size, nz, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = dropout

    def init_state(self, batch_size, spatial_size):
        return self.rnn_p.init_state(batch_size, spatial_size)

    def sample(self, mean, logvar):
        return self.reparam_p.sample_gaussian(mean, logvar)

    def forward(self, prev_state_p, input=None, z=None, higher_hiddens=[]):
        ''' forward stacked rnn one time step
        Input:
            inputs:       a list of inputs, which will be concatenated
            prev_state_p: (hidden, cell) of each ConvLSTM
        Output:
            mean:      batch_size x hidden_size x hidden_height x hidden_width
            logvar:    batch_size x hidden_size x hidden_height x hidden_width
            hidden:    hidden (of new_state), batch_size x hidden_size x hidden_height x hidden_width
            new_state: (hidden, cell) of each ConvLSTM
        '''

        # unpack hidden_p
        hidden_p = prev_state_p[0]

        # compute prior
        input_reparam_p = torch.cat([hidden_p]+higher_hiddens, dim=1)
        mean_p, logvar_p = self.reparam_p(input_reparam_p)

        # sample z
        if z is None:
            z = self.reparam_p.sample_gaussian(mean_p, logvar_p)

        # init input
        input_p = torch.cat(
                [z]
                + higher_hiddens
                + ([input] if input is not None else []),
                dim=1)

        # apply dropout
        ''' see https://discuss.pytorch.org/t/lstm-dropout-clarification-of-last-layer/5588 '''
        if self.dropout > 0:
            input_p = F.dropout(input_p, p=self.dropout, training=self.training, inplace=False)

        # update prior rnn
        hidden_p, new_state_p = self.rnn_p(input_p, prev_state_p)

        return z, mean_p, logvar_p, hidden_p, new_state_p

class StackedConvDrawDecoderCell(nn.Module):
    def __init__(self,
                 #input_size,
                 context_size,  # i.e. nc_context
                 hidden_size,  # i.e. nc_lstm
                 nz,
                 kernel_size=5,
                 padding=2,
                 num_layers=1,
                 dropout=0,
                 ):
        super().__init__()
        #self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.nz = nz
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers
        self.dropout = dropout

        rnns = []
        rnns.append(
                #ConvDrawDecoderCell(input_size + nz + context_size + hidden_size*(num_layers-1),
                ConvDrawDecoderCell(nz + context_size + hidden_size*(num_layers-1),
                                    hidden_size*num_layers,
                                    hidden_size,
                                    nz,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    dropout=self.dropout,
                                    ))
        for i in range(1, num_layers):
            rnns.append(
                    ConvDrawDecoderCell(nz + context_size + hidden_size*(num_layers-(i+1)),
                                        hidden_size*(num_layers-i),
                                        hidden_size,
                                        nz,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        dropout=0 if i == (num_layers-1) else self.dropout,
                                        ))
        self.rnns = nn.ModuleList(rnns)

    def init_state(self, batch_size, spatial_size):
        states = []
        for i in range(len(self.rnns)):
            state = self.rnns[i].init_state(batch_size, spatial_size)
            states += [state]
        return states

    def sample(self, means, logvars):
        zs = []
        for i, (mean, logvar) in enumerate(zip(means, logvars)):
            z = self.rnns[i].sample(mean, logvar)
            zs += [z]
        return zs

    def forward(self, context, prev_states, latents_q=None):
        ''' forward stacked rnn one time step⋅
        Input:
            input: batch_size x input_size
            prev_states: a list with length = number of layers (of stacked RNNs)
                     each element in the list: (h_t, c_t) (or h_t) of each RNN
        Output:
            output_t: batch_size x output_size
            new_states: a list with length = number of layers (of stacked RNNs)
                         each element in the list: (h_t, c_t) (or h_t) of each RNN
        '''
        # init new_states
        new_latents = []
        new_means   = []
        new_logvars = []
        new_hiddens = []
        new_states  = []

        # remaining layers
        for j in range(self.num_layers-1, 0, -1):
            # init input
            z = latents_q[j] if latents_q is not None else None

            # forward rnn
            z_p, mean_p, logvar_p, hidden_p, state = self.rnns[j](
                    prev_states[j], input=context, z=z, higher_hiddens=new_hiddens)
            new_latents += [z_p]
            new_means   += [mean_p]
            new_logvars += [logvar_p]
            new_hiddens += [hidden_p]
            new_states  += [state]

        # init input (last layer)
        z = latents_q[0] if latents_q is not None else None

        # forward rnn (last layer)
        z_p, mean_p, logvar_p, hidden_p, state = self.rnns[0](
                prev_states[0], input=context, z=z, higher_hiddens=new_hiddens)
        new_latents += [z_p]
        new_means   += [mean_p]
        new_logvars += [logvar_p]
        new_hiddens += [hidden_p]
        new_states  += [state]

        # reverse list order
        new_latents = new_latents[::-1]
        new_means   = new_means[::-1]
        new_logvars = new_logvars[::-1]
        new_hiddens = new_hiddens[::-1]
        new_states  = new_states[::-1]

        return new_latents, new_means, new_logvars, new_hiddens, new_states
