import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.label_len > 0:
            print(f'Label length is greater than 0 - no gradients will be taken for the first {configs.label_len} steps.')
        self.pred_len = configs.pred_len
        self.x_in = configs.x_in
        self.y_dim = configs.y_dim + self.x_in * configs.x_dim     # inserting x with y
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.lstm_layers = configs.time_layers
        self.ff_layers = configs.ff_layers
        self.n_draws = configs.n_draws

        # LSTM
        self.lstms = nn.ModuleList([
            nn.LSTMCell(input_size=self.y_dim if l == 0 else self.d_model,
                        hidden_size=self.d_model)
            for l in range(self.lstm_layers)])

        # linear layer
        self.linear = nn.Sequential()
        for ll in range(self.ff_layers):
            self.linear.add_module(f'layer_{ll}', nn.Linear(self.d_model * 2 + self.y_dim if ll == 0 else self.d_ff, self.d_ff))
            self.linear.add_module(f'activition_{ll}', nn.ELU())
        self.linear.add_module(r'layer_out', nn.Linear(self.d_ff, self.c_out))

        self.prev_state = None # [None] * self.lstm_layers

    def forward(self, y, y_tilde, x=None):
        assert x is not None if self.x_in else True, 'x_in is true but no batch x found.'
        assert self.n_draws > 0, 'n_draws must be a positive number.'

        y_inp = y
        if self.x_in:
            y_inp = torch.cat((y_inp, x), dim=-1)

        # init states:
        states = self.prev_state if self.prev_state is not None else \
                 [[tuple([torch.zeros((y.size(0), self.d_model), device=y.device)] * 2) for _ in range(self.lstm_layers)]]

        # input from the original distribution
        outs = []
        for t in range(y_inp.size(1)):
            x_t = y_inp[:, t]
            new_states = []
            for l, lstm in enumerate(self.lstms):
                hx, cx = lstm(x_t, states[-1][l])
                new_states.append((hx, cx))
                x_t = hx
            states.append(new_states)
            out = self.linear(torch.cat([y_inp[:, t], *states[t][-1]], axis=-1)) # last timestep last layer
            outs.append(out)
        outs = torch.stack(outs, dim=1)

        # input from the drawn distribution
        outs_tilde = []
        for i in range(self.n_draws):
            y_tilde_inp = self.draw_y(y, y_tilde)
            if self.x_in:
                y_tilde_inp = torch.cat((y_tilde_inp, x), dim=-1)
            outs_tilde_draw = []
            for t in range(y_inp.size(1)):
                # no need for the lstm to run again - it's only for the original distribution
                out_tilde = self.linear(torch.cat([y_tilde_inp[:, t], *states[t][-1]], axis=-1))
                # using the last layer's state for the linear layer
                outs_tilde_draw.append(out_tilde)
            outs_tilde.append(torch.stack(outs_tilde_draw, dim=1))
        outs_tilde = torch.stack(outs_tilde)
        # saving the last state for next time
        self.prev_state = [[tuple(s.detach() for s in layer_states) for layer_states in states[-1]]]

        return outs[..., -self.pred_len:, :], outs_tilde[..., -self.pred_len:, :]  # [B, L, D]

    def predict_t(self, y, x, y_tilde=None):
        y_inp = y
        if self.x_in:
            y_inp = torch.cat((y_inp, x), dim=-1)

        # init states:
        states = self.prev_state if self.prev_state is not None else \
                 [[tuple([torch.zeros((y.size(0), self.d_model), device=y.device)] * 2) for _ in range(self.lstm_layers)]]

        # input from the original distribution
        outs = []
        for t in range(y_inp.size(1)):
            x_t = y_inp[:, t]
            new_states = []
            for l, lstm in enumerate(self.lstms):
                hx, cx = lstm(x_t, states[-1][l])
                new_states.append((hx, cx))
                x_t = hx
            states.append(new_states)
            out = self.linear(torch.cat([y_inp[:, t], *states[t][-1]], axis=-1)) # last timestep last layer
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        # saving the last state for next time
        self.prev_state = [[tuple(s.detach() for s in layer_states) for layer_states in states[-1]]]

        return outs[:, -self.pred_len:, :]

    @staticmethod
    def draw_y(y, data_min_max):
        return torch.FloatTensor(y.size()).uniform_(*data_min_max).to(y.device)

    def erase_states(self):
        self.prev_state = None
