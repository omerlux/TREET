import torch
import torch.nn as nn
from models.Generators import GeneratorLSTM, GeneratorTransformer, GeneratorTransformerSequence


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.ndg_info = configs.channel_ndg_info['ndg']
        self.x_dim = configs.x_dim
        self.y_dim = configs.y_dim
        self.seq_len = configs.pred_len + configs.label_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len #  * 2 if self.ndg_info['model'] == 'Transformer_Decoder' else configs.label_len
        self.feedback = configs.channel_ndg_info['channel_feedback']
        self.memory_cut = configs.channel_ndg_info['memory_cut']

        self.noise = self.ndg_info['noise']  # gaussian or uniform
        self.zero_mean = self.ndg_info['constraint_zero_mean']
        self.constraint_value = torch.tensor(self.ndg_info['constraint_value'])
        self.constraint_type = self.ndg_info['constraint_type']

        # self.time_layers = self.ndg_info['time_layers']
        self.ndg_model_name = self.ndg_info['model'] if 'model' in self.ndg_info else 'LSTM'
        if self.ndg_model_name == 'LSTM':
            self.ndg_model = GeneratorLSTM(configs)
        elif self.ndg_model_name == 'Decoder_Model':
            self.ndg_model = GeneratorTransformer(configs)
        elif self.ndg_model_name == 'Decoder_Model_Sequence':
            self.ndg_model = GeneratorTransformerSequence(configs)

        self.noise_dim = 1 #(configs.x_dim + configs.y_dim if self.feedback else configs.x_dim) if self.ndg_info['model'] != 'LSTM' else 1

        self.last_x = None
        if self.feedback:
            self.last_y = None

    def forward(self, channel, batch_size=32):
        # states = self.prev_state
        x = self.last_x if self.last_x is not None else \
            torch.zeros((batch_size, self.x_dim), device=self.get_device())
        if self.feedback:
            y = self.last_y if self.last_y is not None else \
                torch.zeros((batch_size, self.y_dim), device=self.get_device())

        batch_x = []
        batch_y = []

        if self.ndg_model_name == 'Decoder_Model_Sequence':
            # noise generation
            u = self._generate_noise(batch_size, self.label_len + self.pred_len)
            x_output = self.ndg_model(inputs=None, noise=u)
            # constraint on output - mean
            if self.zero_mean:
                x_output = self._zero_mean_constraint(x_output)
            # constraint on output - by given type
            if self.constraint_type == 'norm':
                x_output = self._norm_constraint(x_output)  # x'' = x' / sqrt(x'^2) * sqrt(P)

            # apply channel
            for t in range(x_output.size(1)):
                batch_y.append(channel(x_output[:, t]))
            batch_x = x_output
            batch_y = torch.stack(batch_y, dim=1)
        else:
            for t in range(self.label_len + self.pred_len):
                # noise generation
                u = self._generate_noise(batch_size)

                # concatenate inputs - noise and if feedback exists, y
                if self.feedback:
                    inputs = torch.cat((x, y), dim=-1)
                else:
                    inputs = x

                # NDG forward
                x_output = self.ndg_model(inputs, u)

                # constraint on output - mean
                if self.zero_mean:
                    x_output = self._zero_mean_constraint(x_output)
                # constraint on output - by given type
                if self.constraint_type == 'norm':
                    x_output = self._norm_constraint(x_output)          # x'' = x' / sqrt(x'^2) * sqrt(P)

                # apply channel
                y_output = channel(x_output)

                x = x_output.detach()
                if self.feedback:
                    y = y_output.detach()

                batch_x.append(x_output)
                batch_y.append(y_output)

            batch_x = torch.stack(batch_x[-self.seq_len:], dim=1)
            batch_y = torch.stack(batch_y[-self.seq_len:], dim=1)

        # save last states and y if channel feedback exists
        self.ndg_model.detach_states()
        self.last_x = x.detach()
        if self.feedback:
            self.last_y = y.detach()

        return batch_x, batch_y

    def _generate_noise(self, batch_size, seq_len=1):
        size = (batch_size, seq_len, self.noise_dim) if seq_len > 1 else (batch_size, self.noise_dim)
        # Generate a batch of Gaussian noise inputs
        if self.noise == 'gaussian':
            u = torch.randn(*size).to(self.get_device())          # N(0,1)
        else:   # noise == 'uniform'
            u = -1 + 2 * torch.rand(*size).to(self.get_device())  # U[-1,1]
        return u.detach()

    def _zero_mean_constraint(self, x):
        return x - x.mean(dim=0)

    def _norm_constraint(self, x):
        # power constraint on the output \E[x^2] <= P
        eps = 1e-8
        factor = x.pow(2).sum(dim=-1).mean()
        factor = torch.max(factor, torch.tensor(eps, device=self.get_device()))
        return x / factor.sqrt() * self.constraint_value.sqrt()

    def _abs_constraint(self, x):
        # absolute constraint on the output of the ndg
        return x

    def erase_states(self):
        self.ndg_model.erase_states()
        # self.prev_state = [None] * self.time_layers     # states back
        self.last_x = None
        if self.feedback:
            self.last_y = None

    def get_device(self):
        return self.ndg_model.get_device()



