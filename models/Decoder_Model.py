import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Dec import Decoder, DecoderLayer, ConvLayer
from layers.SelfAttention_Family import FullFixedTimeCausalConstructiveAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_temp, FixedEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        assert configs.label_len >= 0 and configs.pred_len > 0, 'Label length is non negative and pred length is positive'
        if configs.label_len > 0:
            print(f'Label length is greater than 0 - no gradients will be taken for the first {configs.label_len} steps.')
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.x_in = configs.x_in

        # Basic:
        self.y_dim = configs.y_dim + self.x_in * configs.x_dim    # inserting x with y
        self.emb_out = configs.d_model
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        self.n_heads = configs.n_heads
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.ff_layers = configs.ff_layers
        self.n_draws = configs.n_draws

        # Embedding
        self.dec_embedding = DataEmbedding_wo_temp(self.y_dim,
                                                   self.emb_out,
                                                   configs.dropout)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    nn.ModuleList([
                        AttentionLayer(
                            FullFixedTimeCausalConstructiveAttention(True, configs.factor, attention_dropout=configs.dropout,
                                                      output_attention=True if l == configs.time_layers - 1 else False,
                                                      history_len=configs.label_len),
                            self.d_model, configs.n_heads)
                        for l in range(configs.time_layers)]),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    ff_layers=configs.ff_layers
                )   # for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, configs.c_out, bias=True)
        )

    def forward(self, y, y_tilde, x=None):
        assert x is not None if self.x_in else True, 'x_in is true but no batch x found.'
        assert self.n_draws > 0, 'n_draws must be a positive number.'

        y_inp = torch.cat((y, x), dim=-1) if self.x_in else y

        dec_out = self.dec_embedding(y_inp)
        dec_out, attns = self.decoder(dec_out, x_mask=None, drawn_y=False)

        dec_outs_tilde = []
        for i in range(self.n_draws):
            y_tilde_inp = self.draw_y(y, y_tilde)
            y_tilde_inp = torch.cat((y_tilde_inp, x), dim=-1) if self.x_in else y_tilde_inp

            dec_out_tilde = self.dec_embedding(y_tilde_inp)
            dec_out_tilde, attns_tilde = self.decoder(dec_out_tilde, x_mask=None, drawn_y=True)

            dec_outs_tilde.append(dec_out_tilde)
        dec_outs_tilde = torch.stack(dec_outs_tilde)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], dec_outs_tilde[..., -self.pred_len:, :], attns, attns_tilde
        else:
            return dec_out[:, -self.pred_len:, :], dec_outs_tilde[..., -self.pred_len:, :]  # [B, L, D]

    def predict_t(self, y, x, y_tilde=None):
        with torch.no_grad():
            if self.x_in:
                assert x is not None
                y = torch.cat((y, x), dim=-1)

            dec_out = self.dec_embedding(y)
            dec_out, attns = self.decoder(dec_out, attn_mask=None, drawn_y=False)

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]

    @staticmethod
    def draw_y(y, data_min_max):
        return torch.FloatTensor(y.size()).uniform_(*data_min_max).to(y.device)
