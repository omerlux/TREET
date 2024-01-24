import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Dec import Decoder, DecoderLayer, ConvLayer
from layers.SelfAttention_Family import FullFixedTimeCausalConstructiveAttention, AttentionLayer, FullAttention
from layers.Embed import DataEmbedding, DataEmbedding_wo_temp, FixedEmbedding


class GeneratorLSTM(nn.Module):

    def __init__(self, configs):
        super(GeneratorLSTM, self).__init__()
        self.ndg_info = configs.channel_ndg_info['ndg']
        self.noise_dim = 1  # configs.x_dim
        self.x_dim = configs.x_dim
        self.y_dim = configs.y_dim

        self.feedback = configs.channel_ndg_info['channel_feedback']

        self.hidden_dim = self.ndg_info['d_model']
        self.d_ff = self.ndg_info['d_ff']
        self.time_layers = self.ndg_info['time_layers']
        self.ff_layers = self.ndg_info['ff_layers']

        # LSTM
        self.lstms = nn.ModuleList([
            nn.LSTMCell(input_size=(self.noise_dim + self.x_dim + self.y_dim if self.feedback else self.noise_dim + self.x_dim) if l == 0 else self.hidden_dim,
                        hidden_size=self.hidden_dim) for l in range(self.time_layers)])

        # linear layer
        layer_list = []
        for i in range(self.ff_layers):
            layer_list.append(nn.Linear(self.d_ff if i else self.hidden_dim, self.d_ff))
            layer_list.append(nn.ELU())
        layer_list.append(nn.Linear(self.d_ff, self.x_dim))
        self.linears = nn.Sequential(*layer_list)

        # for param in self.parameters():
        #     # param.data.normal_(0, 0.05)
        #     torch.nn.init.normal_(param, mean=0, std=.5)

        self.prev_state = [None] * self.time_layers

    def forward(self, inputs, noise):
        states = self.prev_state
        inputs = torch.cat((inputs, noise), dim=-1)

        # propagate states and inputs in RNN layers
        for l in range(self.time_layers):
            hz, cz = self.lstms[l](inputs, states[l])
            states[l] = (hz, cz)
            inputs = hz

        # propagate the output  from RNN to linear layers
        ndg_output = self.linears(inputs)
        self.prev_state = states
        return ndg_output

    def erase_states(self):
        self.prev_state = [None] * self.time_layers

    def detach_states(self):
        self.prev_state = [tuple(s.detach() for s in layer_states) for layer_states in self.prev_state]

    def get_device(self):
        return self.linears[0].weight.device


class GeneratorTransformer(nn.Module):

    def __init__(self, configs):
        super(GeneratorTransformer, self).__init__()
        self.ndg_info = configs.channel_ndg_info['ndg']
        self.feedback = configs.channel_ndg_info['channel_feedback']

        self.noise_dim = 1  # configs.x_dim
        self.x_dim = configs.x_dim
        self.y_dim = configs.y_dim
        self.pred_len = 1
        self.seq_len = self.pred_len + configs.label_len if configs.model != 'LSTM' else configs.pred_len + configs.label_len

        self.d_model = self.ndg_info['d_model']
        self.d_ff = self.ndg_info['d_ff']
        self.time_layers = self.ndg_info['time_layers']
        self.ff_layers = self.ndg_info['ff_layers']

        # # Embedding
        # self.dec_embedding = DataEmbedding_wo_temp(self.x_dim + self.y_dim if self.feedback else self.x_dim,
        #                                            self.d_model,
        #                                            configs.dropout)
        #
        # # Decoder
        # self.decoder = Decoder_for_NDG(
        #     [
        #         DecoderLayer_for_NDG(
        #             nn.ModuleList([
        #                 AttentionLayer(
        #                     FullAttention(True, configs.factor, attention_dropout=configs.dropout,
        #                                   output_attention=True if l == configs.time_layers - 1 else False),
        #                     self.d_model, configs.n_heads)
        #                 for l in range(configs.time_layers)]),
        #             self.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #             ff_layers=configs.ff_layers,
        #             noise_dim=self.noise_dim
        #         )
        #     ],
        #     norm_layer=torch.nn.LayerNorm(self.d_model),
        #     projection=nn.Linear(self.d_model, self.x_dim, bias=True)
        # )


        # Embedding
        self.dec_embedding = DataEmbedding_wo_temp(self.noise_dim + self.x_dim + self.y_dim if self.feedback else self.noise_dim + self.x_dim,
                                                   self.d_model,
                                                   configs.dropout)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    nn.ModuleList([
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=True if l == configs.time_layers - 1 else False),
                            self.d_model, configs.n_heads)
                        for l in range(configs.time_layers)]),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    ff_layers=configs.ff_layers,
                )
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.x_dim, bias=True)
        )

        self.prev_inputs = None
        self.erase_states()

    def forward(self, inputs, noise):
        if inputs.size(0) != self.prev_inputs.shape[0]:
            # Repeat values over batch dimension (dim=0)
            self.prev_inputs = self.prev_inputs.repeat(inputs.size(0), 1, 1).to(self.get_device())

        # Update previous inputs
        self.prev_inputs = torch.cat((self.prev_inputs[:, 1:],
                                      inputs[:, None, :]), dim=1)

        # Each input represents a single timestep
        inputs = torch.cat((self.prev_inputs,
                            torch.zeros_like(inputs[:, None, :])), dim=1)

        # concat noise only to last step
        noise = torch.cat((torch.zeros_like(self.prev_inputs)[..., :self.noise_dim],
                           noise[:, None, :]), dim=1)
        inputs = torch.cat((inputs, noise), dim=-1)

        emb_out = self.dec_embedding(inputs)
        ndg_output, attns = self.decoder(emb_out)

        # Detach previous inputs
        self.prev_inputs = self.prev_inputs.detach().clone()
        return ndg_output[:, -self.pred_len:, :].squeeze(1)
        # return ndg_output

    def erase_states(self):
        self.prev_inputs = torch.zeros((1, self.seq_len - 1,
                                        self.x_dim + self.y_dim if self.feedback else self.x_dim), requires_grad=False)

    def detach_states(self):
        pass

    def get_device(self):
        return self.dec_embedding.value_embedding.weight.device


class GeneratorTransformerSequence(nn.Module):

    def __init__(self, configs):
        super(GeneratorTransformerSequence, self).__init__()
        self.ndg_info = configs.channel_ndg_info['ndg']
        self.feedback = configs.channel_ndg_info['channel_feedback']
        assert self.feedback is False, 'feedback channel is not supported in this model'

        self.noise_dim = 1  # configs.x_dim
        self.x_dim = configs.x_dim
        self.y_dim = configs.y_dim
        self.pred_len = configs.pred_len + configs.label_len
        self.seq_len = configs.pred_len + configs.label_len

        self.d_model = self.ndg_info['d_model']
        self.d_ff = self.ndg_info['d_ff']
        self.time_layers = self.ndg_info['time_layers']
        self.ff_layers = self.ndg_info['ff_layers']

        # Embedding
        self.dec_embedding = DataEmbedding_wo_temp(self.noise_dim,
                                                   self.d_model,
                                                   configs.dropout)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    nn.ModuleList([
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=True if l == configs.time_layers - 1 else False),
                            self.d_model, configs.n_heads)
                        for l in range(configs.time_layers)]),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    ff_layers=configs.ff_layers,
                )
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.x_dim, bias=True)
        )

        self.prev_inputs = None
        self.erase_states()

    def forward(self, inputs, noise):
        emb_out = self.dec_embedding(noise)
        ndg_output, attns = self.decoder(emb_out)

        # Detach previous inputs
        return ndg_output[:, -self.pred_len:, :].squeeze(1)

    def erase_states(self):
        pass

    def detach_states(self):
        pass

    def get_device(self):
        return self.dec_embedding.value_embedding.weight.device
