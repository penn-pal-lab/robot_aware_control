import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class LSTM(nn.Module):
    """
    Vanilla LSTM with embedding layer and output
    """
    name="lstm"

    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                )
            )
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        # pass it through the LSTM cells, and cache hidden states for future
        # forward calls.
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class GaussianLSTM(nn.Module):
    """
    Outputs latent mean and std P(z | x)
    """
    name="gaussian_lstm"

    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(GaussianLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)]
        )
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                )
            )
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM
    """
    def __init__(self, in_ch, hid_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.gates = nn.Conv2d(
            in_ch + hid_ch,
            4*hid_ch,
            kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, input_, prev_state):
        prev_hidden, prev_cell = prev_state
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        out_gates = self.gates(stacked_inputs)
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = out_gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class NormConvLSTMCell(nn.Module):
    """
    Convolutional LSTM
    """
    def __init__(self, in_ch, hid_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.ih_gates = nn.Sequential(
            nn.Conv2d(in_ch, 4*hid_ch, kernel_size, padding=padding, stride=stride),
            nn.GroupNorm(16, 4*hid_ch),
        )

        self.hh_gates = nn.Sequential(
            nn.Conv2d(hid_ch, 4*hid_ch, kernel_size, padding=padding, stride=stride),
            nn.GroupNorm(16, 4*hid_ch),
        )
        self.c_norm = nn.GroupNorm(16, hid_ch)

    def forward(self, input_, prev_state):
        prev_hidden, prev_cell = prev_state
        # data size is [batch, channel, height, width]

        ih_gates = self.ih_gates(input_)
        hh_gates = self.hh_gates(prev_hidden)
        out_gates = ih_gates + hh_gates

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = out_gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        cell = self.c_norm(cell)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvLSTM(nn.Module):
    def __init__(self, config, hid_ch):
        super().__init__()

        self.hid_ch = hid_ch
        Cell = NormConvLSTMCell if config.lstm_group_norm else ConvLSTMCell
        self.lstm = nn.ModuleList(
            [
                Cell(hid_ch, hid_ch, 5, 2, 1),
                Cell(hid_ch, hid_ch, 3, 1, 1),
            ]
        )
        self.batch_size = config.batch_size
        self._width = config.image_width
        self._height = config.image_height
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        """ initializes conv weights for hidden and cell states
        Hidden state shape should be input shape (before conv)
        Cell state shape should be output shape (after conv)
        Args:
            batch_size ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: List of Weights
        """

        if batch_size is None:
            batch_size = self.batch_size
        hidden = []

        n_layers = len(self.lstm)
        b = batch_size
        channels = [self.hid_ch] * n_layers
        hid_heights = [self._height // 8, self._height // 8]
        hid_widths = [self._width // 8, self._width // 8]

        cell_heights = [self._height // 8, self._height // 8]
        cell_widths = [self._width // 8, self._width // 8]

        for i in range(n_layers):
            c, h, w = channels[i], hid_heights[i], hid_widths[i]
            h_state = Variable(torch.zeros(b, c, h, w, device=device))

            c, h, w = channels[i], cell_heights[i], cell_widths[i]
            c_state = Variable(torch.zeros(b, c, h, w, device=device))
            hidden.append((h_state, c_state))

        return hidden

    def forward(self, input_):
        h_in = input_
        for i in range(len(self.lstm)):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        return h_in


class GaussianConvLSTM(ConvLSTM):
    def __init__(self, config, hid_ch, out_ch):
        """Goes from hid_ch -> z dimension

        Args:
            config (): [description]
            hid_ch (int): input channel
        """
        super().__init__(config, hid_ch)
        self.out_ch = out_ch

        # input is (hid_ch, 8, 8) feature map from ConvLSTM
        # output is (out_ch, 8, 8) feature map
        self.mu_net = nn.Conv2d(hid_ch, out_ch, 3, 1, 1)
        self.logvar_net = nn.Conv2d(hid_ch, out_ch, 3, 1, 1)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input_):
        h_in = super().forward(input_)
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class RobonetConvLSTM(nn.Module):
    def __init__(self, batch_size, hid_ch):
        super().__init__()

        self.hid_ch = hid_ch
        self.lstm = nn.ModuleList(
            [
                ConvLSTMCell(hid_ch, hid_ch, 5, 2, 1),
                ConvLSTMCell(hid_ch, hid_ch, 3, 1, 1),
            ]
        )
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        """ initializes conv weights for hidden and cell states
        Hidden state shape should be input shape (before conv)
        Cell state shape should be output shape (after conv)
        Args:
            batch_size ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: List of Weights
        """

        if batch_size is None:
            batch_size = self.batch_size

        hidden = []

        n_layers = len(self.lstm)
        b = batch_size
        channels = [self.hid_ch] * n_layers
        hid_heights = [8, 8]
        hid_widths = [8, 8]

        cell_heights = [8, 8]
        cell_widths = [8, 8]

        for i in range(n_layers):
            c, h, w = channels[i], hid_heights[i], hid_widths[i]
            h_state = Variable(torch.zeros(b, c, h, w, device=device))

            c, h, w = channels[i], cell_heights[i], cell_widths[i]
            c_state = Variable(torch.zeros(b, c, h, w, device=device))
            hidden.append((h_state, c_state))

        self.hidden = hidden

        # T x B x
        self.prev_encs = []

        return hidden

    def forward(self, input_):
        h_in = input_
        batch_size = h_in.shape[0]
        # first pass through 1st ConvLSTM Cell to get attention
        self.hidden[0] = self.lstm[0](h_in, self.hidden[0])
        # (B, 512, 8, 8)
        enc_out = self.hidden[0][0]
        # (B, 32768)
        flatten_enc = enc_out.view(enc_out.shape[0], -1)
        # if first timestep
        if len(self.prev_encs) == 0:
            # T x B x 32768
            self.prev_encs = [flatten_enc]
            attention_enc = enc_out
        else:
            self.prev_encs.append(flatten_enc)
            # T x B x 1
            dot_prods = [torch.sum(flatten_enc * x, 1, keepdims=True) for x in self.prev_encs]
            #cat (T x B x 1, 1) =>  B x T
            attention_weights = F.softmax(torch.cat(dot_prods, 1), dim=1)

            # Run attention over previous encodings (SDP Magic!)
            # ((B x T x 1) * (B x T x 32768)) => (B x 32768)
            attention_enc = torch.sum(attention_weights[:, :, None] * torch.cat([p[:, None] for p in self.prev_encs], 1), 1)
            # (B, 6, 8, 512)
            attention_enc = torch.reshape(attention_enc, [batch_size, self.hid_ch, 8, 8])

        # Pass attention encoding to 2nd ConvLSTM
        self.hidden[1] = self.lstm[1](attention_enc, self.hidden[1])
        h_in = self.hidden[1][0]
        return h_in

if __name__ == "__main__":
    import ipdb

    # conv lstm needs input to have 32 channels
    T = 10
    BS = 16
    G_DIM = 512
    conv_lstm = RobonetConvLSTM(batch_size=BS, hid_ch=G_DIM).to(device)
    dummy_data = torch.ones(T, BS, G_DIM, 8, 8, device=device) # T x B x C x H x W

    conv_lstm.init_hidden(BS)
    for t in range(dummy_data.shape[0]):
        pred_t = conv_lstm(dummy_data[t])
        print(pred_t.shape)