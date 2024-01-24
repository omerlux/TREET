import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, attentions, d_model, d_ff=None,
                 dropout=0.1, activation="relu", ff_layers=1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attentions = attentions

        self.conv1 = nn.Conv1d(in_channels=2 * d_model, out_channels=d_ff, kernel_size=1)
        if ff_layers > 1:
            self.convs = nn.ModuleList([nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=1) for _ in range(ff_layers-1)])
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(attentions) + 1)])
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, x_mask=None, drawn_y=False):
        input = x
        for self_attention, norm_layer in zip(self.attentions, self.norms):
            new_x, attn = self_attention(x, x, x, attn_mask=x_mask, drawn_y=drawn_y)
            x = x + self.dropout(new_x)
            x = norm_layer(x)

        y = torch.cat([input, x], dim=-1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        if hasattr(self, "convs"):
            for conv in self.convs:
                y = self.dropout(self.activation(conv(y)))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norms[-1](y), attn


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, x_mask=None, drawn_y=False):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, x_mask=x_mask, drawn_y=drawn_y)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, attns
